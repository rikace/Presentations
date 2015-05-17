#load "Utilities\AsyncHelpers.fs"
#load "Utilities\FSharp.Charting.fsx"
//#load "Utilities\show-wpf40.fsx"
#r "FSharp.PowerPack.dll"
open System
open System.Net
open System.Text
open System.IO
open System.Drawing
open Microsoft.FSharp.Control.WebExtensions
open System.Windows.Forms.DataVisualization.Charting
open FSharp.Charting

let symbols = [ "MSFT"; "GOOG"; "EBAY"; "AAPL"; "ADBE" ]

let sc (chart:ChartTypes.GenericChart) = chart.ShowChart()

/// Calculates the variance of a sequence
let variance(values:seq<float>) =
    values
    |> Seq.map (fun x -> (1.0 / float (Seq.length values)) * (x - (Seq.average values)) ** 2.0)
    |> Seq.sum

/// Calculates the SD
let stddev(values:seq<float>) =
    values    
    |> Seq.fold (fun acc x -> acc + (1.0 / float (Seq.length values)) * (x - (Seq.average values)) ** 2.0) 0.0
    |> sqrt

let constructURL(symbol, fromDate:DateTime, toDate:DateTime) =
    let formatZero (n:int) = String.Format("{0:00}", n)
    let fm = formatZero(fromDate.Month-1)
    let fd = formatZero(fromDate.Day)
    let fy = formatZero(fromDate.Year)
    let tm = formatZero(toDate.Month-1)
    let td = formatZero(toDate.Day)
    let ty = formatZero(toDate.Year)
    "http://ichart.finance.yahoo.com/table.csv?s=" + symbol + "&d=" + tm + "&e=" + td + "&f=" + ty + "&g=d&a=" + fm + "&b=" + fd + "&c=" + fy + "&ignore=.csv"

let fetchFileAsync (filePath: string) =  async{ 
        try
            use! stream = File.AsyncOpenRead(filePath)
            use reader = new StreamReader(stream)
            let! data = reader.AsyncReadToEnd()
            printfn "Fetching historical data for %s, recieved %d characters" filePath data.Length
            return Some(data)
        with
            | :? System.IO.FileNotFoundException as e -> 
                        printfn "Exception! %s " e.Message
                        return None }
                        
/// Async fetch of CSV data
let fetchAsync(name, url:string) =
    async { 
        try 
            let uri = new System.Uri(url)
            let webClient = new WebClient()
            let! html = webClient.AsyncDownloadString(uri)
            printfn "Downloaded historical data for %s, recieved %d characters" name html.Length
            return Some(html)
        with
            | ex -> printfn "Exception: %s" ex.Message
                    return None }

let getMaxPrice(data:string) =   
    let rows = data.Split('\n')
    rows
    |> Seq.skip 1
    |> Seq.map (fun s -> s.Split(','))
    |> Seq.map (fun s -> float s.[4])    
    |> Seq.take (rows.Length - 2)

let parse (data: string) symbol f = 
    let prices = data.Split('\n')
                                |> Seq.skip 1
                                |> Seq.map (fun s -> s.Split(','))
                                |> Seq.filter (fun s -> Array.length s >= 4)
                                |> Seq.map (fun s -> ((DateTime.Parse s.[0]), 
                                                        f( (s.[1..4]) |> Seq.map float)) )
    (symbol, prices)

let getStocks f fromDate toDate symbol = async {
    let url = constructURL(symbol, fromDate, toDate)
    let! data = fetchAsync(symbol, url)
    match data with 
    | Some(data) -> return Some(parse data symbol f)
    | _ -> return None}


//(getStocks stddev (DateTime.Parse "01/01/2000") (DateTime.Parse "01/01/2014") "MSFT")
//|> Async.RunSynchronously |> show


let getPricesByFile f symbol = async {
    let file = Path.Combine(__SOURCE_DIRECTORY__ ,"Data", symbol + ".csv")
    use! stream = File.AsyncOpenRead(file)
    use reader = new StreamReader(stream)
    let! data = reader.AsyncReadToEnd()
    return parse data symbol f}

let priceFuncs = [ stddev; variance;  ]


let drawChartWebSync f = 
    (symbols
    |> List.map (getStocks f (DateTime.Parse "01/01/2000") (DateTime.Parse "01/01/2014") )
    |> Async.Parallel
    |> Async.RunSynchronously
    |> Array.filter(fun x -> x.IsSome)
    |> Array.map(fun x -> let (symbol, prices) = x.Value
                          Chart.Line(prices, Name=symbol))) |> Chart.Combine |> sc

let drawChartFileSync f = 
    (symbols
    |> List.map (getPricesByFile f)
    |> Async.Parallel
    |> Async.RunSynchronously
    |> Array.map(fun x -> let (symbol, prices) = x
                          Chart.Line(prices, Name=symbol))) |> Chart.Combine |> sc

let drawChartFile f = 
    symbols
    |> List.map (getPricesByFile f)
    |> Async.Parallel

let drawChartWeb f = 
   symbols
    |> List.map (getStocks f (DateTime.Parse "01/01/2000") (DateTime.Parse "01/01/2014") )
    |> Async.Parallel
   

let drawChartFileAsync(f) =
        Async.StartWithContinuations((drawChartFile f),
        (fun result -> result 
                       |> Array.map(fun x -> let (symbol, prices) = x
                                             Chart.Line(prices, Name=symbol))
                                             |> Chart.Combine |> sc),
        (fun _ -> ()), //exception
        (fun _ -> ())) //cancellation


let cancellationSource = new System.Threading.CancellationTokenSource()
let drawChartWebAsync(f) =
        Async.StartWithContinuations((drawChartWeb f),
        (fun result -> result
                       |> Array.filter(fun x -> x.IsSome)
                       |> Array.map(fun x -> let (symbol, prices) = x.Value
                                             Chart.Line(prices, Name=symbol))
                                             |> Chart.Combine |> sc),
        (fun _ -> ()), //exception
        (fun _ -> printfn "The task has been canceled."), cancellationSource.Token) //cancellation

drawChartWebAsync stddev

async { do! Async.Sleep 1000
        cancellationSource.Cancel()
        return () } |> Async.Start

drawChartWeb stddev
drawChartWeb variance

drawChartFile stddev
drawChartFile variance

drawChartWebAsync stddev 
drawChartWebAsync variance


// Reference the Excel interop assemblies
#r @"Microsoft.Office.Interop.Excel.dll"
#r @"office.dll"

open Microsoft.Office.Interop.Excel
open System

let app = new ApplicationClass(Visible = true) 
let workbook = app.Workbooks.Add(XlWBATemplate.xlWBATWorksheet) 
let worksheet = (workbook.Worksheets.[1] :?> _Worksheet) 


worksheet.Range("C2", "E2").Value2 <- [| "2000"; "2005"; "2010" |]


let stocks = let stock = (getStocks stddev (DateTime.Parse "01/01/2000") (DateTime.Parse "01/01/2014") "MSFT")
                         |> Async.RunSynchronously
             stock.Value

let statsArray = (snd stocks) |> Array.ofSeq  

let names = Array2D.init statsArray.Length 1 (fun i _ -> 
  let name, _ = statsArray.[i]
  name )

let dataArray = Array2D.init statsArray.Length 3 (fun x y ->   
  let _, values = statsArray.[x]
  values)

let endColumn = string(statsArray.Length + 2)
worksheet.Range("B3", "B" + endColumn).Value2 <- names
worksheet.Range("C3", "E" + endColumn).Value2 <- dataArray


let chartobjects = (worksheet.ChartObjects() :?> ChartObjects) 
let chartobject = chartobjects.Add(400.0, 20.0, 550.0, 350.0) 

chartobject.Chart.ChartWizard
  (Title = "MSFT Stocs",
   Source = worksheet.Range("B2", "E" + endColumn),
   Gallery = XlChartType.xl3DColumn, PlotBy = XlRowCol.xlColumns,
   SeriesLabels = 1, CategoryLabels = 1,
   CategoryTitle = "", ValueTitle = "Stocks")

chartobject.Chart.ChartStyle <- 13
