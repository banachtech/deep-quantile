module utils_load_crypto

using CSV, DataFrames, Distributions, StatsBase, InMemoryDatasets
using DataStructures, LinearAlgebra
using HTTP, JSON3, Dates
using Flux, Statistics, MultivariateStats, Random
using BSON: @load


"""
    load_data(hp)

Load the data into a dataframe. Number of days of data is equal to obs.
"""

function load_data(hp)
    

    df = hp["data"]
        
    @load "crypto"*"model.bson" model
    obs = Int(size(Flux.params(model)[1])[2]/(size(df)[2]-1))

    df = df[end-obs+1:end,:]        
    
#     df = df[end-hp["obs"]-hp["hold"]+1:end,:]   
#     df = df[end-hp["obs"]+1:end,:]

    return df, obs, model
end

"""
    make_data(df, hp)

Prepare the feature matrix and returns matrix to be input into NN model.
"""

function make_data(df, obs)

    hold = 5
    DataFrames.select!(df, Not(:Date))

    ndate, ncols = size(df)
    ntickers = sum(occursin.("ret", names(df)))

    x_len = obs * ncols

    xs = zeros(Float32, x_len)
    nottickers = ncol(df)-2*ntickers



    xarr = Array(df[1:obs,:])
#     yarr = df[obs+1:end,:]
    
    xs = reshape(xarr, prod(size(xarr)))
#     ys = prod(Array(yarr[:,ntickers+1:end-nottickers]) .+ 1., dims = 1) .- 1.
#     ys = reshape(ys, ntickers)
#     return xs, ys
    return xs
end

"""
    select_tickers(hp)

Select the top and bottom tickers according to quantile reg model
"""

function pred(hp)
    df, obs, model = load_data(hp)
    xs = make_data(df, obs)

    yp = model(xs)
#     idx = sortperm(yp, rev=true)

    tickers = ["BTC", "ETH"]

    
# #     ticks = CSV.read("../data/constituents_csv.csv", DataFrame)
# #     sectors = String.(unique(ticks[!, "Sector"]))
# #     filter!(x->x!=="Utilities", sectors)
# #     for i in 1:length(sectors)
# #         t1 = filter(row -> row.Sector == sectors[i] , ticks)
# #         t = t1[!, "Symbol"]
# #         tickers[sectors[i]]=t
# #     end
    
    n_top = 2
#     n_bottom = 2

#     top = idx[1:n_top]
#     bottom = idx[end-n_bottom+1:end]
#     filtered_yp_top = yp[top]
#     filtered_yp_bottom = yp[bottom]
    
#     top_tick, bottom_tick = tickers[file][top], tickers[file][bottom]
    
    model_pred = Dict()
    for i in 1:n_top
        model_pred[tickers[i]] = yp[i]
    end
#     bottom_d = Dict()
#     for i in 1:length(bottom)
#         bottom_d[bottom_tick[i]] = filtered_yp_bottom[i]
#     end
    
    
    
    return model_pred

end

end