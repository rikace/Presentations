module StockMarket

open System
open System.Collections.Generic
open System.Web
open System.Threading
open Microsoft.AspNet.SignalR
open Microsoft.AspNet.SignalR.Hubs
open System.Reactive.Subjects
open System.Reactive.Linq
open StockTicker.Core
open StockTicker.Server
open FSharp.Collections.ParallelSeq
open TradingAgent
open TradingSupervisorAgent


//  Main "Controller" that run the Stock Market
type StockMarket (?initStocks : Stock seq) =
    let initStocks = defaultArg initStocks Seq.empty<Stock>

    let tradingSupervisorAgent = new TradingSupervisorAgent()


    // fake service-provider that tell "Stock-Ticker" that
    // is time to update the prices
    // I could use a real service... but probably at this time is closed
    // and it is good enough
    let startTicker (stockAgent : Agent<StockTickerMessage>) =
        let rxTimer =
            Observable.Interval(TimeSpan.FromMilliseconds 100.0)
            |> Observable.subscribe(fun _ -> stockAgent.Post UpdateStockPrices)
        rxTimer

    // Agent resposible to upadte the stocks
    // open and close the market and dispatch the orders
    let stockAgent =
        Agent<StockTickerMessage>.Start(fun inbox ->
            let rec marketIsOpen (stocks : ResizeArray<Stock>) (stockTicker : IDisposable) =
                async {
                    let! msg = inbox.Receive()
                    match msg with
                    | GetMarketState(c, reply) ->  reply.Reply(MarketState.Open)
                                                   return! marketIsOpen stocks stockTicker

                    | GetAllStocks(c, reply) -> reply.Reply(stocks |> Seq.toList)
                                                return! marketIsOpen stocks stockTicker

                    | StockTickerMessage.UpdateStock(stock) ->
                        let isStockChanged = updateStocks stock stocks
                        match isStockChanged with
                        | Some(updatedStocks) ->    tradingSupervisorAgent.OnUpdateStock(stock)
                                                    return! marketIsOpen updatedStocks stockTicker
                        | None -> return! marketIsOpen stocks stockTicker

                    | UpdateStockPrices ->
                        for stock in stocks do
                            inbox.Post(StockTickerMessage.UpdateStock(stock))
                        return! marketIsOpen stocks stockTicker

                    | PublishCommand(connId, command) ->
                        match command.Command with
                        | TradingCommand.BuyStockCommand(connId, trading) ->
                            let buy = Buy(connId, trading.Symbol,  {Quantity=trading.Quantity; Price=trading.Price; TradingType = "Buy"})
                            tradingSupervisorAgent.OnNotify(buy)

                        | TradingCommand.SellStockCommand(connId, trading) ->
                            let sell = Sell(connId, trading.Symbol,  {Quantity=trading.Quantity; Price=trading.Price; TradingType = "Sell"})
                            tradingSupervisorAgent.OnNotify(sell)

                        | TradingCommand.AddStockCommand(connId, trading) ->
                                            if not <| (stocks |> Seq.exists (fun s -> s.Symbol.ToUpper() = trading.Symbol.ToUpper())) then
                                                let stock = Stock.CreateStock (trading.Symbol.ToUpper()) trading.Price (stocks.Count)
                                                stocks.Add(stock)
                                                tradingSupervisorAgent.OnNotify(AddStock(connId, stock))
                        return! marketIsOpen stocks stockTicker

                    | CloseMarket(c) ->
                        stockTicker.Dispose()
                        return! marketIsClosed stocks

                    | _ -> return! marketIsOpen stocks stockTicker }

            and marketIsClosed (stocks : ResizeArray<Stock>) = async {
                let! msg = inbox.Receive()
                match msg with
                | GetMarketState(c, reply) ->  reply.Reply(MarketState.Closed)
                                               return! marketIsClosed stocks

                | GetAllStocks(c,reply) -> reply.Reply((stocks |> Seq.toList))
                                           return! marketIsClosed stocks

                | OpenMarket(c) -> return! marketIsOpen stocks (startTicker (inbox))

                | _ -> return! marketIsClosed stocks }
            marketIsClosed (new ResizeArray<Stock>(initStocks)))


    member x.GetAllStocks(connId) =
        stockAgent.PostAndReply(fun ch -> GetAllStocks(connId, ch))

    member x.GetMarketState(connId) =
        stockAgent.PostAndReply(fun ch -> GetMarketState(connId, ch))

    member x.OpenMarket(connId) =
        stockAgent.Post(OpenMarket(connId))

    member x.CloseMarket(connId) =
        stockAgent.Post(CloseMarket(connId))

    member x.UpdateStock(stock:Stock) =
        stockAgent.Post(StockTickerMessage.UpdateStock(stock))

    member x.PublishCommand(connId, order:CommandWrapper) =
        stockAgent.Post(PublishCommand(connId, order))

    member x.Subscribe(connId, initialAmount, caller:IHubCallerConnectionContext<IStockTickerHubClient>) =
        tradingSupervisorAgent.Subscribe(connId, initialAmount, caller) |> ignore

    member x.Unsubscribe(connId) =
        tradingSupervisorAgent.Unsubscribe(connId)

// Singleton for StockTicker
and SingletonStockMarket() =
    static let instanceStockMarket = Lazy.Create(fun () -> StockMarket(Stock.InitialStocks()))
    static member InstanceStockMarket() = instanceStockMarket.Value