module utils_load_data_crypto

using CSV, DataFrames, Distributions, StatsBase, InMemoryDatasets
using DataStructures, LinearAlgebra
using HTTP, JSON3, Dates
using Statistics, Random

##################################################################################################################  
## 1: Functions to scrape and process data 

"""
    get_daily_data(symbol)

Scrapes daily returns and volume data for a given ticker.
"""
function get_daily_data(symbol)

    url1 = "https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_DAILY&symbol="
    url2 = "&market=USD&apikey=0JA1NVBAL8CHYSCH"
    u = url1 * symbol * url2

    resp = HTTP.request("GET", u)
    rawdata = JSON3.read(String(resp.body))
    information = [i for i in rawdata[Symbol("Time Series (Digital Currency Daily)")]]
    format = DateFormat("y-m-d")
    dates = [Date(String(information[i][1]),format) for i in 1:length(rawdata[Symbol("Time Series (Digital Currency Daily)")])]
    adjclose = [parse(Float64,information[i][2][Symbol("4a. close (USD)")]) for i in 1:length(rawdata[Symbol("Time Series (Digital Currency Daily)")])]
    volume = [parse(Float64,information[i][2][Symbol("5. volume")]) for i in 1:length(rawdata[Symbol("Time Series (Digital Currency Daily)")])]

    returns = [(adjclose[i]-adjclose[i+1])./adjclose[i+1] for i in 1:length(adjclose)-1]
    volume = [(volume[i]-volume[i+1])./volume[i+1] for i in 1:length(volume)-1]

    df = DataFrame(Date = dates[1:end-1], returns = returns, volume = volume)
    col_names = ["Date", "ret_"*symbol, "vol_"*symbol]
    
    DataFrames.rename!(df, col_names)


    return sort!(df,[:Date])
end


"""
    combined_data()
scrapes daliy price and volume data of BTC and ETH using the function get_daily_data(symbol), and merges them into a single dataframe

"""


function combined_data()
    tickers = ["BTC", "ETH"]

    df = DataFrame()
    for j in tickers
        try
            d = get_daily_data(j)
            if isempty(df)
                df = d
            else
                df = sort!(outerjoin(df, d, on=:Date, makeunique=true) , [:Date])
            end
        catch
            println("did not get ", j)
        end
    sleep(rand(2:5))
    end

    odd = [i for i in 1:ncol(df) if i%2==1]
    even = [i for i in 1:ncol(df) if i%2==0]

    dfodd = df[!, odd]
    dfeven = df[!, even]
    df = hcat(dfodd, dfeven)
    return df
end
    
end
    
    
