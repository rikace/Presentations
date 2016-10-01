namespace StockAnalyzer.FS

#if INTERCATIVE
#load @"..\src\Common\Utilities\Helpers\FSharp.Charting.fsx"
#endif
open System.Net.Http
open System.Net
open System.IO
open System
open System.Drawing
open FSharp.Charting
open System.Threading
open AsyncEx

[<AutoOpen>]
module Helpers =

    type StockData = {date:DateTime;open':float;high:float;low:float;close:float}
    type StockHistory = {symbol:string;stockData:StockHistory[]}

    let stocks = ["MSFT";"FB";"AAPL";"YHOO"; "EMC"; "AMZN"; "EBAY"; "INTC"; "GOOG"; "ORCL"; "SSY"] //; "CSCO" ]
    


    let chartSymbol symbol (data:seq<DateTime * float * float * float * float>) = 
            let (_,_,max,_,_) = data |> Seq.maxBy(fun (_,_,m,_,_) -> m)  
            let (_,_,_,min,_) = data |> Seq.minBy(fun (_,_,_,m,_) -> m)
            Chart.Candlestick(data, Name=symbol).WithYAxis(Max = max, Min = min) 

    let chartSymbolsAsync (data:Async<(string * seq<DateTime * float * float * float * float>)[]>)= async {
            let! data = data
            let (_,_,max,_,_) = data |> Seq.map snd |> Seq.concat |> Seq.maxBy(fun (_,_,m,_,_) -> m)  
            let (_,_,_,min,_) = data |> Seq.map snd |> Seq.concat |> Seq.minBy(fun (_,_,_,m,_) -> m)

            return Chart.Combine
              [ for s, d in data -> chartSymbol s d ]
                |> Chart.WithArea.AxisY
                    ( Minimum = min, Maximum = max ) 
                |> Chart.WithLegend() }

    let chartSymbols (stockHistories:(string * seq<DateTime * decimal * decimal * decimal * decimal>) seq)=  
            let (_,_,max,_,_) = stockHistories |> Seq.map snd |> Seq.concat |> Seq.maxBy(fun (_,_,m,_,_) -> m)  
            let (_,_,_,min,_) = stockHistories |> Seq.map snd |> Seq.concat |> Seq.minBy(fun (_,_,_,m,_) -> m)

            Chart.Combine
              [ for symbol, prices in stockHistories -> 
                Chart.Candlestick(prices, Name=symbol).WithYAxis(Max = float max, Min = float min) ]
                |> Chart.WithArea.AxisY
                    ( Minimum = float min, Maximum = float max ) 
                |> Chart.WithLegend() 



    let splitData (data:string) = data.Split([|'\n'; '\r'|], StringSplitOptions.RemoveEmptyEntries)

    let getStockHistoryAsync (rows:string[]) fromYear = async{
            // Date (MM-DD-YYYY),Open,High,Low,Close,Volume
            return  rows
                    |> Seq.skip 1
                    |> Seq.map(fun (row:string) -> row.Split([|','|]))
                    |> Seq.filter(fun cells -> DateTime.Parse(cells.[0]).Year > fromYear)
                    |> Seq.map(fun cells ->  DateTime.Parse(cells.[0]).Date,
                                             decimal(cells.[1]), decimal(cells.[2]),
                                             decimal(cells.[3]), decimal(cells.[4])) }


    let getStockHistory (rows:string[]) fromYear = 
            // Date (MM-DD-YYYY),Open,High,Low,Close,Volume
            rows
            |> Seq.skip 1
            |> Seq.map(fun (row:string) -> row.Split([|','|]))
            |> Seq.filter(fun cells -> DateTime.Parse(cells.[0]).Year > fromYear)
            |> Seq.map(fun cells ->  DateTime.Parse(cells.[0]).Date,
                                        decimal(cells.[1]), decimal(cells.[2]),
                                        decimal(cells.[3]), decimal(cells.[4])) 


        
    let googleSourceUrl symbol = sprintf "http://www.google.com/finance/historical?q=%s&output=csv" symbol
    let yahooSourceUrl symbol = sprintf "http://ichart.finance.yahoo.com/table.csv?s=%s" symbol


    // standardDeviation [ for r in msftData.Data -> float r.Close ]
    let standardDeviation prices = 
        let count = Seq.length prices
        let avg = Seq.average prices
        let squares = [ for p in prices -> (p - avg) * (p - avg) ]
        sqrt ((Seq.sum squares) / (float count)) 


module Sequential = 
    open FSharp.Data
    
    type Stocks = CsvProvider<"http://www.google.com/finance/historical?q=msft&output=csv">
    
    let downloadStockHistory fromYear symbol  =    
        let url = sprintf "http://ichart.finance.yahoo.com/table.csv?s=%s" symbol
        let request = HttpWebRequest.Create(url)
        printfn "Downloading %s - Thread Id %d" symbol (Thread.CurrentThread.ManagedThreadId) 
        use respone = request.GetResponse()
        use reader = new StreamReader(respone.GetResponseStream())
        let csvData = reader.ReadToEnd()
        let csvDataTyped = Stocks.Parse(csvData)
        let prices = query { for p in csvDataTyped.Rows do
                                where (p.Date.Year >= fromYear)
                                select (p.Date,p.Open,p.High,p.Low,p.Close) }
        (symbol, prices)

    ServicePointManager.DefaultConnectionLimit <- stocks.Length 

    let analyze() = 
        stocks
        |> List.map(fun symbol -> downloadStockHistory 2010 symbol) 
        |> chartSymbols
    
    
    let chartP = analyze() 
    chartP.ShowChart() |> ignore



module Asynchronous = 

    // In canellcation Token use Start |> Dispoable
    ServicePointManager.DefaultConnectionLimit <- stocks.Length 

    let downloadStockPricesAsync fromYear (sourceUrl:string -> string) symbol = async {        
        let url = sourceUrl symbol
        let request = HttpWebRequest.Create(url) 
        printfn "Downloading %s - Thread Id %d" symbol (Thread.CurrentThread.ManagedThreadId) 
        use! response = request.AsyncGetResponse()
        use reader = new StreamReader(response.GetResponseStream())
        let! data = reader.AsyncReadToEnd()
        let! prices = getStockHistoryAsync (splitData data) fromYear  
        return (symbol, prices) }


    let chartParallel() =
        stocks
        |> Seq.map (fun s -> downloadStockPricesAsync 2010 yahooSourceUrl s)        
        |> Async.Parallel    
        |> Async.RunSynchronously  // ok only for console app becase block a better solutino is Start wth continuation
        |> chartSymbols


    let chartP = chartParallel()
    chartP.ShowChart() |> ignore



















//
//
//
//
//
//
//
//
//module Sequential = 
//
//    let downloadStockHistory fromYear symbol  =    
//        let url = sprintf "http://www.google.com/finance/historical?q=%s&output=csv" symbol
//        let request = HttpWebRequest.Create(url) 
//        use respone = request.GetResponse()
//        use reader = new StreamReader(respone.GetResponseStream())
//        let data = reader.ReadToEnd()
//        let prices = getStockHistory (splitData data) fromYear  
//        (symbol, prices)
//
//    let downloadStockHistory' fromYear (sourceUrl:string -> string) symbol  =    
//        let url = sourceUrl symbol
//        let request = HttpWebRequest.Create(url) 
//        use respone = request.GetResponse()
//        use reader = new StreamReader(respone.GetResponseStream())
//        let data = reader.ReadToEnd()
//        let prices = getStockHistory (splitData data) fromYear  
//        (symbol, prices)
//
//
//    // #time "on"
//    let chartSync() =
//        stocks
//        |> Seq.map (fun s -> downloadStockHistory 2010 s)
//        |> chartSymbols
//
//    let chart = chartSync()
//    chart.ShowChart() |> ignore
//
//
//module AsyncAPM = 
//
//    let fromYear = 2010
//
//    let getResponseCallBack(asyncResult:IAsyncResult) = 
//        let request = asyncResult.AsyncState :?> WebRequest
//        use response = request.EndGetResponse(asyncResult)
//        use reader = new StreamReader(response.GetResponseStream())
//        let data = reader.ReadToEnd()
//        let prices = getStockHistory (splitData data) fromYear  
////        (symbol, prices)
//        ()
//
//    let downloadStockHistory fromYear (sourceUrl:string -> string) symbol  =    
//        let url = sourceUrl symbol
//        let request = HttpWebRequest.Create(url) 
//        request.BeginGetResponse ((fun cb -> getResponseCallBack cb), request)
//
//
//
//    let downloadStockHistoryLamnda fromYear (sourceUrl:string -> string) symbol  =    
//        
//        let asyncResponseCallBack = new AsyncCallback(fun (iar:IAsyncResult) ->
//            let request = iar.AsyncState :?> HttpWebRequest
//            use response = request.EndGetResponse(iar)
//            use reader = new StreamReader(response.GetResponseStream())
//            let data = reader.ReadToEnd()
//            let prices = getStockHistory (splitData data) fromYear  
//            ())
//
//        let url = sourceUrl symbol
//        let request = HttpWebRequest.Create(url) 
//        request.BeginGetResponse (asyncResponseCallBack, request)        
//        
//
////    // #time "on"
////    let chartSync() =
////        stocks
////        |> Seq.map (fun s -> downloadStockHistory 2010 googleSourceUrl s)
////        |> chartSymbols
////
////    let chart = chartSync()
////    chart.ShowChart() |> ignore
//
//
//module Asynchronous = 
//
//
//    // In canellcation Token use Start |> Dispoable
//
//    let setConnectionLimit n = ServicePointManager.DefaultConnectionLimit <- n
//
//    let downloadStockPricesAsync fromYear (sourceUrl:string -> string) symbol = async {
//        printfn "%s initiated on thread %d" symbol Thread.CurrentThread.ManagedThreadId
//        let url = sourceUrl symbol
//        let request = HttpWebRequest.Create(url) 
//        use! response = request.AsyncGetResponse()
//        use reader = new StreamReader(response.GetResponseStream())
//        let! data = reader.AsyncReadToEnd()
//        let! prices = getStockHistoryAsync (splitData data) fromYear  
//        return (symbol, prices) }
//
// 
////    let downloadUrl(url:string) fromYear =
////        let req = HttpWebRequest.Create(url) 
////        async.Bind(req.AsyncGetResponse(), fun rsp ->  // Bind used to compose a workflow returned by AsyncGetResponse with the rest of the computation
////            let rd = new StreamReader(rsp.GetResponseStream()) 
////            async.Bind(rd.AsyncReadToEnd(), fun data ->
////                let data = (splitData data) 
////                async.Bind(getStockHistoryAsync data fromYear, fun res ->  
////                async.Return(res))))
//            
////
////
////    let charting() =
////        stocks
////        |> Seq.map (fun s -> downloadStockPricesAsync 2010 googleSourceUrl s |> Async.RunSynchronously)
////        |> chartSymbols
////            
////    let chartAsync() =
////        stocks
////        |> Seq.map (fun s -> downloadStockPricesAsync 2010 googleSourceUrl s |> Async.RunSynchronously)
////        |> chartSymbols
////
////    let chart = chartAsync()
////    chart.ShowChart() |> ignore
//
//
//    let chartParallel() =
//        stocks
//        |> Seq.map (fun s -> downloadStockPricesAsync 2010 yahooSourceUrl s)        
//        |> Async.Parallel    
//        |> Async.RunSynchronously  // ok only for console app becase block a better solutino is Start wth continuation
//        |> chartSymbols
//
////    let chartParallelCont() =
////        let comp = 
////            stocks
////            |> Seq.map (fun s -> downloadStockPricesAsync 2010 googleSourceUrl s)        
////            |> Async.Parallel    
////            |> chartSymbolsAsync  // input -> output 
////        Async.StartWithContinuations(comp, (fun chart -> chart.ShowChart() |> ignore), (fun exn -> ()), (fun cnl -> ()))
//
//        // Async.StartWithCancelllation(comp, token)
//        // with IDispoable memebr x.Cancel()
//
//    let chartP = chartParallel()
//    chartP.ShowChart() |> ignore
//
//
//
////module chartPlay =
////
////    #if INTERCATIVE
////    #load @"..\src\Common\Utilities\Helpers\FSharp.Charting.fsx"
////    #endif
////    open FSharp.Charting
////    open System.Drawing
////
////    let n = 16
////    let point i =
////        let t = float(i % n) / float n * 2.0 * System.Math.PI in (sin t, cos t)
////
////    Chart.Combine(seq { for i in 0..n-1 do for j in i+1..n -> Chart.Line [point i; point j] })
////    |> Chart.WithYAxis(Enabled=false)
////    |> Chart.WithXAxis(Enabled=false)
////    |> Chart.Show
////
////    open FSharp.Charting
////    open System.Drawing
////    open System.Windows.Forms.DataVisualization.Charting
////
////    let myBackground = ChartTypes.Background.Gradient (Color.LightSkyBlue,Color.White,GradientStyle.DiagonalRight)
////    let myFrame = ChartTypes.Background.Solid Color.Blue
////
////    // drawing per se...
////    Chart.Combine(
////        seq { for x in 0.0..16.0..1024. do
////                    let repers = [x,0.; 512.,x; 512.-x,512.; 0.,512.-x; x,0.] 
////                    for (x0,y0),(xl,yl) in Seq.pairwise repers -> Chart.Line [(x0,y0);(xl,yl)] })
////    // ...the rest is pure styling
////    |> Chart.WithXAxis(Enabled=false)
////    |> Chart.WithYAxis(Enabled=false)
////    |> Chart.WithStyling(AreaBackground=myBackground,Background=myFrame)
////    //|> Chart.Show