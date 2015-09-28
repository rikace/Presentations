open System.IO
open System

let createStockTickerScript() =
    let script = (new StockTickerSignalRClient.Wrapper()).GenerateScript()
    Console.WriteLine script
    File.WriteAllText(@"..\..\..\..\StockTicker\Scripts\stockTickerFS.js", script)

[<EntryPoint>]
let main argv = 
    
    createStockTickerScript()

    0  

