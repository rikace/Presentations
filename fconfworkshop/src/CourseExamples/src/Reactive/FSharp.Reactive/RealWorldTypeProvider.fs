module RealWorldTypeProvider

open FSharp.Data

[<Literal>]
let yahooapis = "https://query.yahooapis.com/v1/public/yql"
[<Literal>]
let apiParameters = "&format=json&env=store://datatables.org/alltableswithkeys"
[<Literal>]
let londonLocationSample = 
    yahooapis + "?q=select woeid from geo.places(1) where text = %22london%22" + apiParameters
[<Literal>]
let londonWeatherSample = 
    yahooapis + "?q=select * from weather.forecast where woeid = 44418" + apiParameters

let locationApi (location : string) = 
    yahooapis + "?q=select woeid from geo.places(1) where text = %22" + location + "%22" + apiParameters
let weatherApi (location : string) = 
    yahooapis + "?q=select * from weather.forecast where woeid = "+ location + apiParameters

type Location = JsonProvider<londonLocationSample>
let location = Location.Load(locationApi "tel aviv").Query.Results.Place.Woeid.ToString()
printfn "Woeid: %s" location

type Weather = JsonProvider<londonWeatherSample> 
let query = Weather.Load(weatherApi location).Query
let units = query.Results.Channel.Units.Temperature
printfn "Temperature %s" units

let forecast = query.Results.Channel.Item.Forecast |> Array.map (fun w -> w.Date, w.Text)
printfn "%A" forecast