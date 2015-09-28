module StockTickerHub
 
open Owin
open Microsoft.Owin
open System
open System.Collections.Generic
open System.Web
open System.Threading
open Microsoft.AspNet.SignalR
open Microsoft.AspNet.SignalR.Hubs
open System.Reactive.Subjects
open System.Reactive.Linq
open StockTickerServer
open FSharp.Collections.ParallelSeq
open TradingAgent 
open TradingSupervisorAgent
open StockMarket
open StockTickerHubClient
        

     // Hub with "Stock Market" operations
     // It is responsible to update the stock-ticker
     // to each client registered
    [<HubName("stockTicker")>]
    type StockTickerHub() as this =        
        inherit Hub<IStockTickerHubClient>()
   
        static let userCount = ref 0
 
        let stockMarket : StockMarket = SingletonStockMarket.InstanceStockMarket()
     
        override x.OnConnected() =
            ignore <| System.Threading.Interlocked.Increment(userCount)
            let connId = x.Context.ConnectionId
        
            // Subscribe a new client
            stockMarket.Subscribe(connId, 1000., this.Clients) // I can use this.Clients.Caller but this is demo purpose
            base.OnConnected()
           
        override x.OnDisconnected(stopCalled) =
            ignore <| System.Threading.Interlocked.Decrement(userCount)
            let connId = x.Context.ConnectionId

            // Unsubscribe client
            stockMarket.Unsubscribe(connId)
            base.OnDisconnected(stopCalled)

        member x.GetAllStocks() =
                let connId = x.Context.ConnectionId
                let stocks = stockMarket.GetAllStocks(connId)
                for stock in stocks do
                    this.Clients.Caller.SetStock stock
 
        member x.OpenMarket() =
                let connId = x.Context.ConnectionId
                stockMarket.OpenMarket(connId)
                this.Clients.All.SetMarketState(string MarketState.Open)
 
        member x.CloseMarket() =
                let connId = x.Context.ConnectionId
                stockMarket.CloseMarket(connId)
                this.Clients.All.SetMarketState(string MarketState.Closed)
