module StockTickerHubClient
 
open System
open Microsoft.AspNet.SignalR
open Microsoft.AspNet.SignalR.Hubs
open StockTickerServer


// interface StockTickerHub SignalR
type IStockTickerHubClient =
    abstract SetMarketState : string -> unit
    abstract UpdateStockPrice : Stock -> unit
    abstract SetStock : Stock -> unit
    abstract UpdateOrderBuy : OrderRecord -> Unit
    abstract UpdateOrderSell : OrderRecord -> Unit
    abstract UpdateAsset : Asset -> Unit
    abstract SetInitialAsset : float -> Unit