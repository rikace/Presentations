namespace StockTicker

open System
open System.Threading

type Agent<'T> = MailboxProcessor<'T>

//RunDll32.exe InetCpl.cpl, ClearMyTracksByProcess 8 - Clear Temp Files
//RunDll32.exe InetCpl.cpl, ClearMyTracksByProcess 2 - Clear Cookies
//RunDll32.exe InetCpl.cpl, ClearMyTracksByProcess 1 - Clear History
//RunDll32.exe InetCpl.cpl, ClearMyTracksByProcess 16 - Clears Form Data
//RunDll32.exe InetCpl.cpl, ClearMyTracksByProcess 32 - Clears Passwords
//RunDll32.exe InetCpl.cpl, ClearMyTracksByProcess 255 - Clears All
//RunDll32.exe InetCpl.cpl, ClearMyTracksByProcess 4351 - Clears Add On Files and Settings

//type VoteCounts = { language : string; count : int }  
//     
//type Message = 
//    | Vote of string * AsyncReplyChannel<seq<string*int>>
//
//let votesAgent = MailboxProcessor.Start(fun inbox ->
//    let rec loop votes =
//        async {
//            let! message = inbox.Receive()
//            match message with
//            | Vote(language, replyChannel) -> 
//                let newVotes = language::votes 
//                newVotes
//                |> Seq.countBy(fun lang -> lang)
//                |> replyChannel.Reply 
//                do! loop(newVotes) 
//            do! loop votes 
//        }
//    loop List.empty)
//
//type ChartHub() =
//    inherit Hub()
//    member x.Send (data:string) = 
//       let result = 
//           votesAgent.PostAndReply(fun reply -> Message.Vote(data, reply))   
//           |> Seq.map(fun v -> { language = fst v; count = snd v } )
//       try
//           base.Clients?updateChart(result)
//       with
//       | ex -> 
//           printfn "%s" ex.Message
//
//let server = Server "http://*:8181/"
//server.MapHubs() |> ignore
//
//server.Start()
//
//printfn "Now listening on port 8181"
//Console.ReadLine() |> ignore
//
//
//open System
//open Newtonsoft.Json
//
//type VoteCounts = { language : string; count : int }  
//     
//type Message = 
//    | Vote of string * AsyncReplyChannel<seq<string*int>>
//
//let votesAgent = MailboxProcessor.Start(fun inbox ->
//    let rec loop votes =
//        async {
//            let! message = inbox.Receive()
//            match message with
//            | Vote(language, replyChannel) -> 
//                let newVotes = language::votes 
//                newVotes
//                |> Seq.countBy(fun lang -> lang)
//                |> replyChannel.Reply 
//                do! loop(newVotes) 
//            do! loop votes 
//        }
//    loop List.empty)
//
//type ChartServer() =
//    inherit PersistentConnection()
//
//    override x.OnReceived(request, connectionId, data) = 
//        votesAgent.PostAndReply(fun reply -> Message.Vote(data, reply))   
//        |> Seq.map(fun v -> { language = fst v; count = snd v } )
//        |> JsonConvert.SerializeObject 
//        |> base.Connection.Broadcast


type Stock = 
    { Symbol : string
      DayOpen : decimal
      DayLow : decimal
      DayHigh : decimal
      LastChange : decimal
      Price : decimal }
    member x.Change = x.Price - x.DayOpen
    member x.PercentChange = double (Math.Round(x.Change / x.Price, 4))
    static member CreateStock symbol price = 
        { Symbol = symbol
          LastChange = 0M
          Price = price
          DayOpen = 0M
          DayLow = 0M
          DayHigh = 0M }
    
    static member InitialStocks() = seq {
            for stock in [ ("MSFT", 41.68m)
                           ("APPL", 92.08m)
                           ("GOOG", 543.01m) ] -> stock ||> Stock.CreateStock }

type MarketState = 
    | Closed
    | Open
    with override x.ToString() = match x with
                                 | Open -> "Open"
                                 | Closed -> "Closed"
        
type StockMessage = 
    | UpdateStock of Stock
    | AddStock of Stock
    | UpdateStockPrices
    | Reset
    | OpenMarket
    | CloseMarket
    | GetMarketState //of AsyncReplyChannel<MarketState>
    | GetAllStocks //of AsyncReplyChannel<Stock seq>




type StockTicker(?initStocks : Stock seq) = 
    let initStocks = defaultArg initStocks Seq.empty<Stock>
    let rnd = new Random()
    
    let onStockUpdates = Event<Stock>()
    let onGetMarketState = Event<MarketState>()
    let onGetAllStocks = Event<Stock list>()
    let onSetMarketState = Event<MarketState>()
    
    let changePriceStock (stock : Stock) (price : decimal) = 
        if price = stock.Price then stock
        else 
            let lastChange = price - stock.Price
            let dayOpen = 
                if stock.DayOpen = 0m then price
                else stock.DayOpen
            let dayLow = 
                if price < stock.DayLow || stock.DayLow = 0m then price
                else stock.DayLow
            let dayHigh = 
                if price > stock.DayHigh then price
                else stock.DayHigh
            { stock with Price = price
                         LastChange = lastChange
                         DayOpen = dayOpen
                         DayLow = dayLow
                         DayHigh = dayHigh }
    
    let updateStock (stock : Stock) = 
        let r = rnd.NextDouble()
        if r > 0.1 then (false, stock)
        else 
            let rnd' = Random(int (Math.Floor(stock.Price)))
            let percenatgeChange = rnd'.NextDouble() * 0.002
            let change = 
                let change = Math.Round(stock.Price * (decimal percenatgeChange), 2)
                if (rnd'.NextDouble() > 0.51) then change
                else -change
            let newStock = changePriceStock stock (stock.Price + change)
            (true, newStock)
    
    let startTicker (stockAgent : MailboxProcessor<StockMessage>) = 
        let timer = new System.Timers.Timer(float 200)
        timer.AutoReset <- true
        let disposable = timer.Elapsed |> Observable.subscribe (fun _ -> stockAgent.Post UpdateStockPrices)
        timer.Start()
        disposable
    
    let stockAgent = 
        MailboxProcessor<_>.Start(fun inbox -> 
            let rec loop (stocks : ResizeArray<Stock>) (stockTicker : IDisposable) = 
                async { 
                    let! msg = inbox.Receive()
                    match msg with
                    | GetMarketState ->  onGetMarketState.Trigger MarketState.Open
                                         return! loop stocks stockTicker
                    | GetAllStocks ->    onGetAllStocks.Trigger (stocks |> Seq.toList)
                                         return! loop stocks stockTicker
                    | AddStock(stock) -> if not <| (stocks |> Seq.exists (fun s -> s.Symbol = stock.Symbol)) then 
                                            stocks.Add(stock)
                                            return! loop stocks stockTicker
                    | UpdateStockPrices -> 
                        for stock in stocks do inbox.Post(UpdateStock(stock))
                        return! loop stocks stockTicker
                    | UpdateStock(stock) -> 
                        let isStockChange = updateStock stock
                        match isStockChange with
                        | true, newStock -> 
                            let index = stocks |> Seq.tryFindIndex (fun s -> s.Symbol = stock.Symbol)
                            match index with
                            | None -> return! loop stocks stockTicker
                            | Some(index) ->    stocks.[index] <- newStock
                                                onStockUpdates.Trigger newStock
                        | false, _ -> return! loop stocks stockTicker
                        return! loop stocks stockTicker
                    | CloseMarket -> 
                        stockTicker.Dispose()
                        onSetMarketState.Trigger MarketState.Closed
                        return! marketIsClosed stocks
                    | _ -> return! loop stocks stockTicker }
            and marketIsClosed (stocks : ResizeArray<Stock>) = async {
                    let! msg = inbox.Receive()
                    match msg with
                    | GetMarketState ->  onGetMarketState.Trigger MarketState.Closed
                                         return! marketIsClosed stocks
                    | GetAllStocks ->    onGetAllStocks.Trigger (stocks |> Seq.toList)
                                         return! marketIsClosed stocks
//                    | Reset ->  stocks.Clear()
//                                clients.All?marketReset ()
                                //return! marketIsClosed stocks
                    | OpenMarket -> onSetMarketState.Trigger MarketState.Open
                                    return! loop stocks (startTicker (inbox))
                    | _ -> return! marketIsClosed stocks
                }
            marketIsClosed (new ResizeArray<Stock>(initStocks)))
    
    member x.GetMarketState() = stockAgent.Post(GetMarketState)
    member x.OpenMarket() = stockAgent.Post OpenMarket
    member x.CloseMarket() = stockAgent.Post CloseMarket
    member x.Reset() = stockAgent.Post Reset
    member x.AddStock symbol price = let stock = Stock.CreateStock symbol price
                                     stockAgent.Post(AddStock(stock))
    member x.GetAllStocks() = stockAgent.Post(GetAllStocks)

    member x.OnStockUpdates = onStockUpdates.Publish
    member x.OnGetMarketState = onGetMarketState.Publish
    member x.OnGetAllStocks = onGetAllStocks.Publish
    member x.OnSetMarketState = onSetMarketState.Publish