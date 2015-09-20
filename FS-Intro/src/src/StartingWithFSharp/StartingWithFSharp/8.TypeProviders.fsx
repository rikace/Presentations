//           _ _                            _                              
//     /\   | | |                          | |                             
//    /  \  | | |  _   _  ___  _   _ _ __  | |_ _   _ _ __   ___  ___      
//   / /\ \ | | | | | | |/ _ \| | | | '__| | __| | | | '_ \ / _ \/ __|     
//  / ____ \| | | | |_| | (_) | |_| | |    | |_| |_| | |_) |  __/\__ \     
// /_/    \_\_|_|  \__, |\___/ \__,_|_|     \__|\__, | .__/ \___||___/     
//                  __/ |                        __/ | |                   
//                 |___/        _               |___/|_|                   
//                  | |        | |                   | |                   
//   __ _ _ __ ___  | |__   ___| | ___  _ __   __ _  | |_ ___    _   _ ___ 
//  / _` | '__/ _ \ | '_ \ / _ \ |/ _ \| '_ \ / _` | | __/ _ \  | | | / __|
// | (_| | | |  __/ | |_) |  __/ | (_) | | | | (_| | | || (_) | | |_| \__ \
//  \__,_|_|  \___| |_.__/ \___|_|\___/|_| |_|\__, |  \__\___/   \__,_|___/
//                                             __/ |                       
//                                            |___/                        

#load "../packages/FSharp.Charting.0.90.12/FSharp.Charting.fsx" 
#load ".\\Common\\show-wpf40.fsx"
#r "System.Data.Services.Client.dll"
#r "System.Runtime.Serialization.dll"
#r "FSharp.Data.TypeProviders.dll"
#r @"..\packages\FSharp.Data.2.2.5\lib\net40\FSharp.Data.dll"
#r "System.ServiceModel"

open FSharp.Charting
open Microsoft.FSharp.Data.TypeProviders
open System
open FSharp.Data
open System.Runtime.Serialization
open System.ServiceModel
open FSharp.Charting
open FSharp.Charting.ChartTypes
open ShowWpf


// Type-Providers are one of the biggest reason to use F# !!!

(*  The type providers for structured formats it infers the structure.
    An F# type provider is a component that provides types, properties, 
    and methods for use in your program.
    
    The types provided by F# type providers are usually based on external information sources. 
    
    XML WMI CSV HTTP WSDL 
    *)


module CSVProvider =
(*  The CSV type provider takes a sample CSV as input and generates a type based on 
    the data present on     the columns of that sample. The column names are obtained 
    from the first (header) row, and the types are inferred from the values present 
    on the subsequent rows.   *)

    type Stocks = CsvProvider<"Data\\MSFT.csv">


    // Download the stock prices
    let msft = Stocks.Load("Data\\MSFT.csv")
    //let msft = Stocks.Load("http://ichart.finance.yahoo.com/table.csv?s=MSFT")

    // Look at the most recent row. Note the 'Date' property
    // is of type 'DateTime' and 'Open' has a type 'decimal'
    let firstRow = msft.Rows |> Seq.head
    let lastDate = firstRow.Date
    let lastOpen = firstRow.Open

    // Print the prices in the HLOC format
    for row in msft.Rows do
      printfn "HLOC: (%A, %A, %A, %A)" row.High row.Low row.Open row.Close

    // Visualize the stock prices
    [ for row in msft.Rows -> row.Date, row.Open ]
    |> Chart.FastLine


    /// Helper method that returns a Yahoo! Finance
    /// URL for specified stock and date range
    let urlFor ticker (startDate:DateTime) (endDate:DateTime) = 
//      let root = "http://ichart.finance.yahoo.com/table.csv"
//      sprintf "%s?s=%s&a=%i&b=%i&c=%i&d=%i&e=%i&f=%i" root ticker 
//              (startDate.Month - 1) startDate.Day startDate.Year 
//              (endDate.Month - 1) endDate.Day endDate.Year
        sprintf "Data\\%s.csv" ticker


    // Helper function that returns dates & closing prices from 2012
    let recentPrices symbol =
      let data = Stocks.Load(urlFor symbol (DateTime(2014,1,1)) DateTime.Now)
      [ for row in (data.Rows |> Seq.filter(fun s -> s.Date.Year >= 2015)) -> row.Date.DayOfYear, row.Close ]

    // Compare two stock prices in a single chart        
    Chart.Combine
      [ Chart.Line(recentPrices "GOOG", Name="Google")
        Chart.Line(recentPrices "MSFT", Name="Microsoft").WithLegend() ]

    Chart.Combine
      [ for symbol in ["MSFT"; "AMZN"; "GOOG"] do
          let data = recentPrices symbol
          yield Chart.Line(data, Name=symbol).WithLegend() ]



module ``JSON TypeProvider`` =

    [<Literal>]
    let schema1 = """{ "Name" : "Riccardo", "Age" : 21 }"""

    [<Literal>]
    let schema2 = """{ "Users" : [ { "Username" : "TestUser", "PercentageComplete" : 4.7 }, { "Username" : "TestUser2", "PercentageComplete" : 97.4 } ] }"""

    type tschema1 = JsonProvider<schema1>
    type tschema2 = JsonProvider<schema2>

    let sample1 = tschema1.Parse(schema1)
    sample1.Name

    let sample2 = tschema2.Parse(schema2)
    for user in sample2.Users do
        printfn "%s: %f" user.Username user.PercentageComplete


    let apiUrl = "http://api.openweathermap.org/data/2.5/weather?q="
    type Weather = JsonProvider<"http://api.openweathermap.org/data/2.5/weather?q=London">

    let sf = Weather.Load(apiUrl + "San Francisco")
    sf.Sys.Country
    sf.Wind.Speed
    sf.Main




    type Json = JsonProvider<"Data/Data.json">


    let getSpendings id =
        Json.Load "Data.json"
        |> Seq.filter (fun c -> c.Id = id)
        |> Seq.collect (fun c -> c.Spendings 
                                 |> Seq.map float)
        |> List.ofSeq

    type Csv = CsvProvider<"Data/Data.csv">

    type Customer = {Id:int;IsVip:bool;Credit:decimal;}

    let getCustomers () = 
        let file = Csv.Load "Data/Data.csv"
        file.Rows
        |> Seq.map (fun c -> 
            { Id = c.Id
              IsVip = c.IsVip
              Credit = c.Credit 
              })
    
/// WSDL ///
let cities =  
    [
    ("Burlington", "VT");
    ("Kensington", "MD");
    ("Panama City Beach", "FL");
    ("Knoxville", "TN");
    ("Chicago", "IL");
    ("Casper", "WY"); 
    ("Denver", "CO");
    ("Phoenix", "AZ"); 
    ("Seattle", "WA");
    ("Los Angeles", "CA"); 
    ]

module CheckAddress = 
    type ZipLookup = 
        WsdlService<ServiceUri = "http://www.webservicex.net/uszip.asmx?WSDL", Async=true>

    let GetZip citySt =
        let city, state = citySt
        let findCorrectState (node:System.Xml.XmlNode) = (state = node.SelectSingleNode("STATE/text()").Value)

        let results = ZipLookup.GetUSZipSoap().GetInfoByCity(city).SelectNodes("Table") 
                        |> Seq.cast<System.Xml.XmlNode> 
                        |> Seq.filter findCorrectState
        (results |> Seq.nth 0).SelectSingleNode("ZIP/text()").Value

    GetZip ("Denver", "CO")

    cities
  //  |> Seq.take 3
    |> Seq.map GetZip
    |> Seq.iter (printfn "%s")

module GetTemps = 
    
    let url = "http://api.openweathermap.org/data/2.5/weather?units=metrics&q="
    Http.RequestString(url + "London,UK")

    type Weather = JsonProvider< "http://api.openweathermap.org/data/2.5/weather?units=metric&q=Redmond">
    
    let lw = Weather.Load(url + "London,UK")
    lw.Sys.Country
    lw.Main.Humidity
    lw.Main.Temp

    let getTemp city =
        let w = Weather.Load(url + city)
        let temp = w.Main.Temp        
        printfn "City %s - Temp %2.2f" city temp // %s ??
        int(temp)

    getTemp "Denver,CO"
    getTemp "Burlington,VT"

    let temp_in cityList = 
        let convertCitiesToZips (cc:string * string) = 
            let city, country = cc
            let zip = CheckAddress.GetZip(cc)
            (sprintf "%s,%s" city country), zip
        List.map convertCitiesToZips cityList

    let data = cities |> temp_in |> List.map(fun (cityCountry, zip) ->  let temp = getTemp cityCountry
                                                                        (cityCountry, zip, string(temp)))

    Chart.Bubble(data, Title="Temperature by Zip", UseSizeForLabel=false).WithYAxis(Enabled=true, Max=100000., Min=0.).WithXAxis(Enabled=true).WithDataPointLabels()   

module WorldBankProvider = 

    let wb = WorldBankData.GetDataContext()

    type WorldBank = WorldBankDataProvider<"World Development Indicators", Asynchronous=true>
    WorldBank.GetDataContext()

    wb
      .Countries.``United Kingdom``
      .Indicators.``School enrollment, tertiary (% gross)``
    |> teeGrid

    wb.Countries.``United Kingdom``
        .Indicators.``School enrollment, tertiary (% gross)``
    |> Chart.Line 


    let countries = 
     [| wb.Countries.``Arab World``
        wb.Countries.``European Union``
        wb.Countries.Australia
        wb.Countries.Brazil
        wb.Countries.Canada
        wb.Countries.Chile
        wb.Countries.``Czech Republic``
        wb.Countries.Denmark
        wb.Countries.France
        wb.Countries.Greece
        wb.Countries.``Low income``
        wb.Countries.``High income``
        wb.Countries.``United Kingdom``
        wb.Countries.``United States`` |]

    // Compare a list of countries
    [ for c in countries -> 
        async { return (c.Indicators.``School enrollment, tertiary (% gross)``, c) }]
    |> Async.Parallel
    |> Async.RunSynchronously
    |> Array.map (fun (data, country) ->  Chart.Line(data, Name=country.Name))
    |> Chart.Combine


     // Calculate average data for all OECD members (Organisation for Economic Co‑operation and Development)
    let oecd = [ for c in wb .Regions.``OECD members``.Countries do
                   yield! c.Indicators.``School enrollment, tertiary (% gross)`` ]
               |> Seq.groupBy fst
               |> Seq.map (fun (y, v) -> y, Seq.averageBy snd v)
               |> Array.ofSeq
 
    let it = wb.Countries.Italy.Indicators.``School enrollment, tertiary (% gross)``
    let us = wb.Countries.``United States``.Indicators.``School enrollment, tertiary (% gross)``
    let ru = wb.Countries.``Russian Federation``.Indicators.``School enrollment, tertiary (% gross)``

    Chart.Combine
       [ Chart.Line(oecd)
         Chart.Line(us)
         Chart.Line(ru)
         Chart.Line(it) ]




    let countries = 
       [ wb.Countries.``El Salvador``
         wb.Countries.China 
         wb.Countries.Malaysia
         wb.Countries.Singapore
         wb.Countries.Germany
         wb.Countries.``United States``
         wb.Countries.India
         wb.Countries.Afghanistan
         wb.Countries.``Yemen, Rep.``
         wb.Countries.Bangladesh ]

    /// Chart the populations, un-normalized
    [ for c in countries -> 
        async { return Chart.Line (c.Indicators.``Population ages 0-14 (% of total)``, Name=c.Name) }]
    |> Async.Parallel
    |> Async.RunSynchronously
    |> (fun res -> Chart.Combine(res).WithTitle("Population, 1960-2012"))


//    Chart.Combine([ for c in countries -> Chart.Line (c.Indicators.``Population ages 0-14 (% of total)``, Name=c.Name) ])
//        .WithTitle("Population, 1960-2012")
//         
    Chart.Pie
       [ for c in countries -> c.Name,  c.Indicators.``Total Population (in number of people)``.TryGetValueAt(2001).Value ]

    // Total Population
    #load @"..\..\..\..\packages\FsLab.0.3.5\FsLab.fsx" 
    #r "Deedle.dll"

    open FsLab
    open Deedle
    open Foogle
    open FSharp.Data

    let world = wb.Countries.Italy.Indicators
    let indCode = world.``Total Population (in number of people)``.IndicatorCode

    let getPage  indCode year page =
        Http.RequestString
            ("http://api.worldbank.org/countries/all/indicators/" + indCode,
                query=[ "format", "json"; "date", sprintf "%d:%d" year year;
                        "per_page", "100"; "page", string page])

    getPage indCode 2010 1

    type WBJson = JsonProvider<"http://api.worldbank.org/countries/all/indicators/SP.POP.TOTL?format=json&date=2000:2000&per_page=100">

    WBJson.Parse(getPage indCode 2010 1)        
    
    let w = WBJson.Parse(getPage indCode 2010 1)        
    w.Record.Total
    w.Array.[0].Country


    let getIndicator year indCode = seq {
        let info = WBJson.Parse(getPage indCode year 1)
        for p in 1 .. info.Record.Pages do
            let page = WBJson.Parse(getPage indCode year p)
            for d in page.Array do
                match d.Value with
                | Some v when v < 1500000000L ->
                    yield d.Country.Value => float v
                | _ -> () }

    let wb2000 = series(getIndicator 2000 indCode)
    let wb2010 = series(getIndicator 2010 indCode)
    Chart.GeoChart(Series.observations wb2000)

    let wbChange = (wb2010 - wb2000) / wb2010 * 100.0
    Chart.GeoChart(Series.observations wbChange)

module NorthWind =

    type NorthWind = ODataService<"http://services.odata.org/V3/Northwind/Northwind.svc">
    let svc = NorthWind.GetDataContext()

    let invoices = query {
        for i in svc.Invoices do
        sortByNullableDescending i.ShippedDate
        take 5
        select (i.OrderDate, i.CustomerName, i.ProductName)  }

    invoices |> Seq.iter(printfn "%A")

