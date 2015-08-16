#r "../../packages/FSharp.Data.2.2.5/lib/Net40/FSharp.Data.dll"
#r "../../packages/FSharp.Data.TypeProviders.0.0.1/lib/net40/FSharp.Data.TypeProviders.dll"
#r "../FSharpLib/bin/Debug/FSharpLib.exe"
#load "../FSharpLib/show-wpf40.fsx"
#r "System.Data.Services.Client.dll"

open ShowWpf
open Show
open FSharp.Data

let apiUrl = "http://api.openweathermap.org/data/2.5/weather?q="
type Weather = JsonProvider<"http://api.openweathermap.org/data/2.5/weather?q=London">

let sf = Weather.Load(apiUrl + "San Francisco")
sf.Sys.Country
sf.Wind.Speed
sf.Main



type NorthWind = Microsoft.FSharp.Data.TypeProviders.ODataService<"http://services.odata.org/V3/Northwind/Northwind.svc">
let svc = NorthWind.GetDataContext()

let invoices = query {
    for i in svc.Invoices do
    sortByNullableDescending i.ShippedDate
    take 5
    select (i.OrderDate, i.CustomerName, i.ProductName)  }

invoices |> Seq.iter(printfn "%A")



#load "../../packages/FSharp.Charting.0.90.12/FSharp.Charting.fsx"
open System
open FSharp.Charting

type Stocks = CsvProvider<"Data/MSFT.csv">
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

