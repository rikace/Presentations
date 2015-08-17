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

#r "System.Data.Services.Client.dll"
#r "System.Runtime.Serialization.dll"
#r "FSharp.Data.TypeProviders.dll"
#r @"..\packages\FSharp.Data.2.2.5\lib\net40\FSharp.Data.dll"
#load "../packages/FSharp.Charting.0.90.12/FSharp.Charting.fsx" 
#r " System.ServiceModel.dll"
#load ".\\Common\\show-wpf40.fsx"

open FSharp.Charting
open Microsoft.FSharp.Data.TypeProviders
open System
open FSharp.Data

// Type-Providers are one of the biggest reason to use F# !!!

(*  The type providers for structured formats it infers the structure.
    An F# type provider is a component that provides types, properties, 
    and methods for use in your program.
    
    The types provided by F# type providers are usually based on external information sources. 
    
    XML WMI CSV HTTP WSDL 
    *)


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





//TYPE PROVIDERS
#r "System.ServiceModel"
#r "System.Runtime.Serialization"

open System.Runtime.Serialization
open System.ServiceModel

/// WSDL ///
let cities =  
    [
    ("Burlington", "VT");
    ("Kensington", "MD");
    ("Port Jefferson", "NY"); 
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
    type ZipLookup = Microsoft.FSharp.Data.TypeProviders.WsdlService<ServiceUri = "http://www.webservicex.net/uszip.asmx?WSDL">

    let GetZip citySt =
        let city, state = citySt
        let findCorrectState (node:System.Xml.XmlNode) = (state = node.SelectSingleNode("STATE/text()").Value)

        let results = ZipLookup.GetUSZipSoap().GetInfoByCity(city).SelectNodes("Table") 
                        |> Seq.cast<System.Xml.XmlNode> 
                        |> Seq.filter findCorrectState
        (results |> Seq.nth 0).SelectSingleNode("ZIP/text()").Value

module GetTemps = 
    type WeatherService = Microsoft.FSharp.Data.TypeProviders.WsdlService<ServiceUri = "http://wsf.cdyne.com/WeatherWS/Weather.asmx?WSDL">
    let weather = WeatherService.GetWeatherSoap().GetCityWeatherByZIP

    let temp_in cityList = 
        let convertCitiesToZips city = 
            let zip = CheckAddress.GetZip city
            ((weather zip).City, zip, (weather zip).Temperature)
        List.map convertCitiesToZips cityList

    let data = temp_in <| cities
    Chart.Bubble(data, Title="Temperature by Zip", UseSizeForLabel=false).WithYAxis(Enabled=true, Max=100000., Min=0.).WithXAxis(Enabled=true).WithDataPointLabels()






open FSharp.Data

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


     // Calculate average data for all OECD members
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





         -----





         
// ----------------------------------------------------------------------------
// Load the charting library

#load "../../../packages/FSharp.Charting.0.90.12/FSharp.Charting.fsx"

open FSharp.Charting
open FSharp.Charting.ChartTypes
 
// ----------------------------------------------------------------------------
// Reference the provider for the World Bank and explore the data

#r "../../../packages/FSharp.Data.2.2.5/lib/Net40/FSharp.Data.dll"
#r "../../../packages/FSharp.Data.TypeProviders.0.0.1/lib/net40/FSharp.Data.TypeProviders.dll"

let data = FSharp.Data.WorldBankData.GetDataContext()

let countries = 
   [ data.Countries.``El Salvador``
     data.Countries.China 
     data.Countries.Malaysia
     data.Countries.Singapore
     data.Countries.Germany
     data.Countries.``United States``
     data.Countries.India
     data.Countries.Afghanistan
     data.Countries.``Yemen, Rep.``
     data.Countries.Bangladesh ]



/// Chart the populations, un-normalized
Chart.Combine([ for c in countries -> Chart.Line (c.Indicators.``Population ages 0-14 (% of total)``, Name=c.Name) ])
     .WithTitle("Population, 1960-2012")




Chart.Pie
   [ for c in countries -> c.Name,  c.Indicators.``Population, total``.TryGetValueAt(2001).Value ]


let defaultZero v =
    match v with
    | None -> 0.
    | Some(n) -> n

let countries1 = 
  [ data.Countries.India; data.Countries.Uganda; data.Countries.Ghana;
    data.Countries.``Burkina Faso``; data.Countries.Niger; data.Countries.Malawi
    data.Countries.Afghanistan; data.Countries.Cambodia; data.Countries.Bangladesh
  ]

let pointdata = 
    [ for country in countries1 ->
          let y = defaultZero (country.Indicators.``Adolescent fertility rate (births per 1,000 women ages 15-19)``.TryGetValueAt(2005))
          let x = defaultZero (country.Indicators.``Primary completion rate, female (% of relevant age group)``.TryGetValueAt(2005))
          x,y ]
                 

Chart.Point(pointdata)
     .WithXAxis(Title="Adolescent fertility rate (births per 1,000 women ages 15-19)")
     .WithYAxis(Title="Primary completion rate, female (% of relevant age group)")
     .WithMarkers(Size=40,Style=MarkerStyle.Diamond)



data.Countries.Australia.Indicators.``Population, total``

// ----------------------------------------------------------------------------
// Work with time series data

//#load "packages/Deedle/Deedle.fsx"


data.Countries.``United States``.Indicators.``Health expenditure, total (% of GDP)``
|> Chart.Line


let countries2 = 
  [ data.Countries.``United States``; data.Countries.Switzerland
    data.Countries.Denmark; data.Countries.``United Kingdom``;
    data.Countries.``Czech Republic`` ]

Chart.Combine([ for country in countries2 ->
                    let data = country.Indicators.``Health expenditure per capita (current US$)``
                    Chart.Line(data, Name=country.Name) ])
     .WithTitle("Health expenditure per capita (current US$)")
     .WithLegend(InsideArea=false)


Chart.Combine([ for country in countries2 ->
                    let data = country.Indicators.``Mortality rate, infant (per 1,000 live births)``
                    Chart.Line(data, Name=country.Name) ])
     .WithTitle("Mortality rate, infant (per 1,000 live births)")
     .WithXAxis(Max=2011.0 (* , TickMarks=[1960..3..2010] *) )
             




















let (++) xs ys = Seq.append xs ys

let popAt year =
 ((query { for c in data.Countries  do 
           let pop = c.Indicators.``Population, total``.GetValueAtOrZero(year) 
           where (pop > 15000000.0) 
           select (c.Name, pop) }
   ++
   [ ("Other", 
      query { for c in data.Countries  do 
              let pop = c.Indicators.``Population, total``.GetValueAtOrZero(year) 
              where (pop < 15000000.0) 
              sumBy (pop) }) ])
   
   |> Seq.sortBy snd)

#load "extlib/AsyncSeq-0.1.fsx"
open FSharp.Charting.ChartTypes

open Samples.FSharp.AsyncSeq
asyncSeq { for year in 1960..10..2009 do yield popAt year; do! Async.Sleep 1000 } 
   |> AsyncSeq.StartAsEvent
   |> LiveChart.Pie
Chart.Pie(popAt 2004,Name="Population")


/// Normalize the time series using the value at the given key as the 1.0 value
let normalize key xs = 
    let firstValue = xs |> Seq.find (fun (y,v) -> y = key) |> snd
    xs |> Seq.map (fun (y,v) -> (y, float v / firstValue) )

/// Test it
data.Countries.Australia.Indicators.``Population, total`` |> normalize 1960 
data.Countries.Australia.Indicators.``Population, total`` |> normalize 1960 |> Chart.Line
data.Countries.Australia.Indicators.``Population, total`` |> normalize 1960 |> Chart.Line

//data.GetCountry("AUS")._GetIndicator("SP.POP.TOTL").GetValueAtOrZero(2001)

[ for c in data.Countries -> c.Code ]
[ for c in data.Regions -> c.Name ]

/// Chart the populations, normalized
Chart.Combine 
    [ for c in countries -> 
        let data = c.Indicators.``Population, total`` |> normalize 1960
        Chart.Line(data, Name=c.Name)]
    |> fun c -> c.WithTitle("Population, Normalized").WithLegend(InsideArea=false)

Chart.Combine 
    [ for c in countries ->
        let data = c.Indicators.``International migrant stock (% of population)``
        Chart.Line (data, Name=c.Name) ]
    |> fun c -> c.WithTitle("International migrants")



Chart.Combine 
    [ for c in countries ->
        let data = c.Indicators.``Malnutrition prevalence, height for age (% of children under 5)``
        Chart.Line (data, Name=c.Name) ]
    |> fun c -> c.WithTitle("Malnutrition for Children under 5, compared")


// ----------------------------------------------------------------------------
// How are we doing on debt?

data.Countries.Greece.Indicators.``Central government debt, total (% of GDP)``
|> Chart.Line


// Plot debt of different countries in a single chart using nicer chart style

let countries4 = 
  [ data.Countries.Greece; data.Countries.Ireland; 
    data.Countries.Denmark; data.Countries.``United Kingdom``;
    data.Countries.``Czech Republic`` ]

Chart.Combine
  [ for country in countries4 ->
      let data = country.Indicators.``Central government debt, total (% of GDP)``
      Chart.Line(data, Name=country.Name) ]
|> fun c -> c.WithTitle("Central government debt, total") //  .WithLegend(Docking = Docking.Left)


// ----------------------------------------------------------------------------
// University enrollment

open System.Drawing
    
/// Calculate average university enrollment for EU
/// (This is slow because it needs to download info for every EU country)
let avgEU =
    [ for c in data.Regions.``European Union``.Countries do
        yield! c.Indicators.``School enrollment, tertiary (% gross)`` ]
    |> Seq.groupBy fst
    |> Seq.map (fun (y, v) -> y, Seq.averageBy snd v)
    |> Array.ofSeq
    |> Array.sortBy fst

/// Calculate average university enrollment for OECD
/// (This is slow because it needs to download info for every OECD country)
let avgOECD =
    [ for c in data.Regions.``OECD members``.Countries do
        yield! c.Indicators.``School enrollment, tertiary (% gross)`` ]
    |> Seq.groupBy fst
    |> Seq.map (fun (y, v) -> y, Seq.averageBy snd v)
    |> Array.ofSeq
    |> Array.sortBy fst

// Generate nice line chart combining CZ, EU and OECD enrollment
Chart.Combine
  [ yield Chart.Line(avgEU, Name="EU", Color=Color.Blue)
    yield Chart.Line(avgOECD, Name="OECD", Color=Color.Goldenrod)
    let cze = data.Countries.``Czech Republic``
    yield Chart.Line(data=cze.Indicators.``School enrollment, tertiary (% gross)``,Name="CZ",Color=Color.DarkRed)  ]
|> fun c -> c.WithLegend(Docking = Docking.Left)





(*

//.WithYAxis(MajorGrid = dashGrid).WithAxisX(MajorGrid = dashGrid)
//|> Chart.WithTitle("Czech, EU and OECD University Enrollments", Color=Color.Black, Docking=Docking.Top)


// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// Other tests....


data.Countries.Australia.Indicators.``Improved water source (% of population with access)``
data.Countries.Afghanistan.Indicators.``Improved water source (% of population with access)``
data.Countries.China.Indicators.``Improved water source (% of population with access)``
Chart.Combine 
    [ for c in countries ->
        let data = c.Indicators.``Improved water source (% of population with access)``
        Chart.Line (data, Name=c.Name) ]
                


data.Countries.France.Indicators.``ARI treatment (% of children under 5 taken to a health provider)``.Source

data.Countries.Afghanistan.Code
data.Countries.Denmark.Indicators.``Debt buyback (current US$)``.Source 
data.Countries.Denmark.Indicators.``Total reserves (% of total external debt)``.Source
data.Countries.Denmark.Indicators.``Short-term debt (% of total reserves)``.Source
data.Countries.Afghanistan.Indicators.``Access to electricity (% of population)``.Source

// ----------------------------------------------------------------------------
// Access the data....

data.Countries.Greece.Indicators.``Land under cereal production (hectares)``.Source
data.Countries.Greece.Indicators.``ARI treatment (% of children under 5 taken to a health provider)``
data.Countries.Afghanistan.Indicators.``Access to electricity (% of population)``
data.Countries.Greece.Indicators.``Access to electricity (% of population)``
data.Countries.Greece.Indicators.``Adjusted net national income (annual % growth)``
data.Countries.Greece.Indicators.``Adjusted net national income (annual % growth)``

data.Countries.Australia.Indicators.``Land area (sq. km)`` 
data.Countries.Australia.Indicators.``Land under cereal production (hectares)``
data.Countries.Australia.Indicators.``Land under cereal production (hectares)``


data.Countries.Greece.Indicators.``Central government debt, total (% of GDP)``

// Area covered by forests in World and different EU countries
//http://api.worldbank.org/regions/WLD/indicators/AG.LND.AGRI.ZS?per_page=1000&date=1900%3a2050&page=1
//http://api.worldbank.org/countries/WLD/indicators/AG.LND.AGRI.ZS?per_page=1000&date=1900%3a2050&page=1
data.Regions.World.Countries
data.Regions.World.Countries.
data.Regions.``European Union``.Countries
data.Regions.World.Indicators.``Access to electricity (% of population)``
data.Regions.World.Indicators.``Agricultural land (% of land area)``

 .Countries.World.``Forest area (% of land area)``
|> Seq.sortBy fst

let euforests = 
  [ for country in WorldBank.Regions.``European Union`` do
      let forests = country.``Forest area (% of land area)`` |> Seq.sortBy fst |> Seq.toArray
      yield country, snd forests.[forests.Length - 1] ]

euforests |> List.sortBy (fun (_, f) -> -f) 

*)
// TODO: add static and dynamic parameters to turn off caching of values
// TODO:  alwaysFresh=true 
// TODO: add TickMarks=[1960..3..2010] to FSharpChart

//type T = Samples.WorldBankDataProvider<"World Development Indicators">



module CSVProvider =
(*  The CSV type provider takes a sample CSV as input and generates a type based on 
    the data present on     the columns of that sample. The column names are obtained 
    from the first (header) row, and the types are inferred from the values present 
    on the subsequent rows.   *)

    type Stocks = CsvProvider<"Common\\MSFT.csv">


    // Download the stock prices
    let msft = Stocks.Load("http://ichart.finance.yahoo.com/table.csv?s=MSFT")

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
      let root = "http://ichart.finance.yahoo.com/table.csv"
      sprintf "%s?s=%s&a=%i&b=%i&c=%i&d=%i&e=%i&f=%i" root ticker 
              (startDate.Month - 1) startDate.Day startDate.Year 
              (endDate.Month - 1) endDate.Day endDate.Year


    // URL used to infer the schema of the CSV file and column types
    [<Literal>]
    let msftUrl = "http://ichart.finance.yahoo.com/table.csv?s=MSFT"

    // Helper function that returns dates & closing prices from 2012
    let recentPrices symbol =
      let data = Stocks.Load(urlFor symbol (DateTime(2014,1,1)) DateTime.Now)
      [ for row in data.Rows -> row.Date.DayOfYear, row.Close ]

    // Compare two stock prices in a single chart        
    Chart.Combine
      [ Chart.Line(recentPrices "AAPL", Name="Apple")
        Chart.Line(recentPrices "MSFT", Name="Microsoft").WithLegend() ]

    Chart.Combine
      [ for symbol in ["MSFT"; "AMZN"; "AAPL"; "GOOG"; "FB"] do
          let data = recentPrices symbol
          yield Chart.Line(data, Name=symbol).WithLegend() ]

module ServiceProvider = 


    // Wheater
    //type TerraService = WsdlService<"http://msrmaps.com/TerraService2.asmx?WSDL">
    type TerraService = WsdlService<"http://wsf.cdyne.com/weatherws/weather.asmx">
    let client = TerraService.GetWeatherSoap12()
    let forecast = client.GetCityForecastByZIP("20745")
    forecast.ForecastResult |> Seq.iter(fun r -> printfn  "%A" r.Temperatures.DaytimeHigh
                                                 showA r.Temperatures.DaytimeHigh)


    /// WSDL ///
    let cities =  
        [
        ("Burlington", "VT");
        ("Kensington", "MD");
        ("Port Jefferson", "NY"); 
        ("Panama City Beach", "FL");
        ("Knoxville", "TN");
        ("Chicago", "IL");
        ("Casper", "WY"); 
        ("Denver", "CO");
        ("Phoenix", "AZ"); 
        ("Seattle", "WA");
        ("Los Angeles", "CA"); 
        ]

    type ZipLookup = Microsoft.FSharp.Data.TypeProviders.WsdlService<ServiceUri = "http://www.webservicex.net/uszip.asmx?WSDL">

    let GetZip citySt =
        let city, state = citySt
        let findCorrectState (node:System.Xml.XmlNode) = (state = node.SelectSingleNode("STATE/text()").Value)

        let results = ZipLookup.GetUSZipSoap().GetInfoByCity(city).SelectNodes("Table") 
                        |> Seq.cast<System.Xml.XmlNode> 
                        |> Seq.filter findCorrectState
        (results |> Seq.nth 0).SelectSingleNode("ZIP/text()").Value

    GetZip ("Casper", "WY")


    
    type WeatherService = Microsoft.FSharp.Data.TypeProviders.WsdlService<ServiceUri = "http://wsf.cdyne.com/WeatherWS/Weather.asmx?WSDL">

    let weather = WeatherService.GetWeatherSoap().GetCityWeatherByZIP

    let temp_in cityList = 
        let convertCitiesToZips city = 
            let zip = GetZip city
            ((weather zip).City, zip, (weather zip).Temperature)
        List.map convertCitiesToZips cityList

    let data = temp_in <| cities
    Chart.Bubble(data, Title="Temperature by Zip", UseSizeForLabel=false).WithYAxis(Enabled=true, Max=10000., Min=0.).WithXAxis(Enabled=true).WithDataPointLabels()

