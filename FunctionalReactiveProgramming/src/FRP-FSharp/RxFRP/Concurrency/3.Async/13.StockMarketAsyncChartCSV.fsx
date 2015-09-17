#load "..\Utilities\AsyncHelpers.fs"
#load "..\Utilities\FSharp.Charting.fsx"

open System
open System.IO
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

// Read File Async
let fetchFileAsync (filePath: string) =  async{ 
        try
            use stream = File.OpenRead(filePath)
            use reader = new StreamReader(stream)
            let! data = reader.ReadToEndAsync() |> Async.AwaitTask
            printfn "Fetching historical data for %s, recieved %d characters" filePath data.Length
            return Some(data)
        with
            | :? System.IO.FileNotFoundException as e -> 
                        printfn "Exception! %s " e.Message
                        return None }

let getPricesByFile f symbol = async {
    let file = System.IO.Path.Combine(System.IO.Path.GetDirectoryName(__SOURCE_DIRECTORY__ ),"Data\\" + symbol + ".csv")
    printfn "File %s Exists %b" file (File.Exists file)
    use stream = File.OpenRead(file)
    use reader = new StreamReader(stream)
    let! data = reader.ReadToEndAsync() |> Async.AwaitTask
    return parse data symbol f}

let priceFuncs = [ stddev; variance;  ]

let drawChartFile f = 
    symbols
    |> List.map (getPricesByFile f) // Seq<Async<string * seq<DateTime * float>>
    |> Async.Parallel

let cancellationSource = new System.Threading.CancellationTokenSource()

let mapSymbolToChart (arr: (string * seq<DateTime * float>)[]) = 
        arr
        |> Array.map(fun x ->   let (symbol, prices) = x
                                Chart.Line(prices, Name=symbol))
        |> Chart.Combine |> sc

// RunSynchronously vs StartWithContinuations

drawChartFile stddev 
    |> Async.RunSynchronously  
    |> mapSymbolToChart

drawChartFile variance
    |> Async.RunSynchronously  
    |> mapSymbolToChart

let drawChartFileAsync(f) = 
        Async.StartWithContinuations((drawChartFile f),
            (mapSymbolToChart),
            (fun _ -> ()), //exception
            (fun _ -> ())) //cancellation


drawChartFileAsync stddev
drawChartFileAsync variance


