module utils

using Flux, CSV, DataFrames, Statistics, MultivariateStats, Random
using BSON: @save

"""
    load_data(hp)

Load the data into a dataframe.
"""
function load_data(hp)
    dir = hp["dir"]
    file = hp["file"]
    df = DataFrame(CSV.File("$(dir)$(file).csv"))
    return df
end

"""
    make_data(df, hp)

Prepare the feature matrix and returns matrix to be input into NN model.
"""
function make_data(df, hp)
    obs = hp["obs"]
    hold = hp["hold"]
    select!(df, Not(:Date))
            
    ndate, ncols = size(df)
    ntickers = sum(occursin.("ret", names(df)))

    x_len = obs * ncols
    
    rng = obs+1:hold:ndate-hold
    T = length(rng)

    xs = zeros(Float32, x_len, T)
    ys = zeros(Float32, ntickers, T)
    
    nottickers = ncol(df)-2*ntickers
    
    for (i, t) in enumerate(rng)
        arr = Array(df[t-obs:t-1,:])
        xs[:,i] = reshape(arr, prod(size(arr)))

        ys[:,i] = prod(Array(df[t+1:t+hold,ntickers+1:end-nottickers]) .+ 1., dims = 1) .- 1.
    end
    ind = findall(x->isnan(x),xs)
    for i in ind
        xs[i[1],i[2]] = xs[i[1]-1,i[2]]
    end
    ind = findall(x->isinf(x),xs)
    for i in ind
        xs[i[1],i[2]] = xs[i[1]-1,i[2]]
    end

    return xs, ys
end

# function make_data(df, hp)
#     # this takes in combined data consisting of returns and volume per sector per stock 
#     obs = hp["obs"]
#     hold = hp["hold"]
#     df = df[:,cld(ncol(df),2):end] # this removes all the volumes data
    
#     # this only selects stocks with data that are >40% non-zero (some are 100% zero for some reason)
#     df2 = DataFrame()
#     for i in 1:ncol(df)
#         if length(findall(x->x==0  ,df[:,i]))/length(df[:,i])<0.4
#             df2 = hcat(df2, df[:,i], makeunique = true)
#         end
#     end
#     df = df2
    
#     nsamples, ncoy = nrow(df), ncol(df)
#     data_df = [df[t-obs-hold+1:t, :] for t = obs+hold:nsamples]
    
#     # generate the higher order moments (features), 
#     # and cumulated returns over a holding period (the answer for supervised learning)
#     returns = zeros(length(data_df), 5, ncol(df))
#     y_returns = zeros(length(data_df), ncol(df)-2)
#     return_df = zeros(5, ncol(df))
            
#     for i in 1:length(data_df)
#         obs_df = data_df[i][1:obs,:]
#         holding_df = data_df[i][obs+1:end-2,:]
#         for j in 1:5
#             if j == 1
#                 return_df[j,:] = [mean(c) for c in eachcol(obs_df)]
#             elseif j == 2
#                 return_df[j,:] = [maximum(c) for c in eachcol(obs_df)]
#             elseif j == 3
#                 return_df[j,:] = [mean((c.-mean(c)).^2) for c in eachcol(obs_df)]
#             elseif j == 4
#                 return_df[j,:] = [mean((c.-mean(c)).^3) for c in eachcol(obs_df)]
#             elseif j == 5
#                 return_df[j,:] = [mean((c.-mean(c)).^4) for c in eachcol(obs_df)]

#             end
#         end
#         y_returns[i,:] = [cumprod(holding_df[:,k].+1)[end].-1 for k in 1:ncol(df)]
#         returns[i, :, :] = return_df

#     end     

#     returns = [vec(returns[i, :, :]) for i in 1:length(data_df)]
    
#     m = zeros(length(returns), length(returns[1])+ncol(df))
#     for i in 1:size(m)[1]
#         m[i,1:length(returns[1])] = returns[i]
#     m[i, length(returns[1])+1:end,:] = y_returns[i,:]
#     end
#     df = DataFrame(m, :auto)

#     xs = df[:,1:end-ncoy]
#     xs = Matrix(transpose(Array(xs)))

#     ys = df[:,end-ncoy+1:end]
#     ys = Matrix(transpose(Array(ys)))

# #     M = fit(PCA, Matrix(xs); pratio=0.9)
# #     xs = predict(M, xs)
#     return xs, ys
# end

"""
    make_dense_model(xs, ys, hp)

Create and return a dense NN model such that input and output dimensions are compatible with features matrix `xs` and dependent variable matrix `ys`.
"""
function make_dense_model(xs, ys, hp)
    input_len = size(xs, 1)
    output_len = size(ys, 1)
    l = []
    Random.seed!(hp["seed"])
    try
        l = hp["denselayers"]
    catch
    end
    layers = [(Dense(l[i], l[i+1]),Flux.Dropout(0.2))  for i in 1:lastindex(l)-1]

    layers = [i for t in layers for i in t]

    layers = (Dense(input_len, l[1], leakyrelu), Flux.Dropout(0.1), layers..., Dense(l[end], output_len))
    
    model = Flux.Chain(layers)
    return model
end

"""
    smad_obj(yp, y, hp)

Calculate the mean squared deviation between predicted values `yp` and true values `ys`, parameterised by the target percentile `τ`.
"""
function smad_obj(yp, y, hp)
    τ = hp["τ"]
    err = y .- yp
    return (mean(max.(τ * err, (τ-1) * err)))^2
end

"""
    evaluate(yp, y)

Calculate the average percentile of predicted values `yp` with respect to true values `y`.
"""
function evaluate(yp, y)
    diff = y - yp
    percentile = sum(diff .< 0, dims=2) ./ size(y, 2)
    return percentile
end

"""
    train(hp)

Train a dense NN model on dataset. Return the results of the training as a named tuple.
"""
function train(hp; benchmark=true)
    optimiser = hp["optimiser"]
    df = load_data(hp)
    xs, ys = make_data(df, hp)
    split = hp["split"]
    t = Int(floor(size(xs, 2)*split))
    xs_train, xs_test = xs[:, 1:t], xs[:, t+1:end]
    ys_train, ys_test = ys[:, 1:t], ys[:, t+1:end]

    m = make_dense_model(xs_train, ys_train, hp)
    opt_state = Flux.setup(optimiser(hp["η"]), m)
    loss(model, x, y) = smad_obj(model(x), y, hp)
    train_hist = []
    test_hist = []
    smad_hist_tr = []
    smad_hist_te = []

    for e in 1:hp["epochs"]
        try
            Flux.train!(loss, m, [(xs_train, ys_train)], opt_state)

            yp_train = m(xs_train)
            yp_test = m(xs_test)

            percentile_train = evaluate(yp_train, ys_train)
            percentile_test = evaluate(yp_test, ys_test)
            smad_tr = smad_obj(yp_train, ys_train, hp)
            smad_te = smad_obj(yp_test, ys_test, hp)


            push!(train_hist, percentile_train)
            push!(test_hist, percentile_test)
            push!(smad_hist_tr, smad_tr)
            push!(smad_hist_te, smad_te)
        catch
            println("ERROR AT EPOCH ", e)
            break
        end
    end
    yp_train = m(xs_train)
    yp_test = m(xs_test)

    percentile_train = evaluate(yp_train, ys_train)
    percentile_test = evaluate(yp_test, ys_test)

    if hp["saveroot"] !== nothing
        @save "$(hp["saveroot"])_model_$(hp["symbol"]).bson" m
    end

    if benchmark == false
        return (xs_train = xs_train, ys_train = ys_train, yp_train = yp_train, pc_train = percentile_train, train_hist = train_hist, xs_test = xs_test, ys_test = ys_test, yp_test = yp_test, pc_test = percentile_test, test_hist = test_hist, model = m)
    elseif benchmark == true
        τ = hp["τ"]
        yp_bench = zeros(Float32, size(yp_test))
        yp_bench_tr = zeros(Float32, size(yp_train))
        val = mapslices(d -> quantile(d, hp["τ"]), ys_train, dims=2)
        #val = quantile!(vec(ys_train), τ)
        yp_bench .= val
        yp_bench_tr .= val
        percentile_bench = evaluate(yp_bench, ys_test)
        percentile_bench_train = evaluate(yp_bench_tr, ys_train)

        return (xs_train = xs_train, ys_train = ys_train, yp_train = yp_train, pc_train = percentile_train, train_hist = train_hist, xs_test = xs_test, ys_test = ys_test, yp_test = yp_test, pc_test = percentile_test, test_hist = test_hist, yp_bench = yp_bench, pc_bench = percentile_bench, pc_bench_train = percentile_bench_train, model = m, smad_tr_hist = smad_hist_tr, smad_te_hist = smad_hist_te)
    end
end

end
