
[<AutoOpenAttribute>]
module TradingAgent

open System
open System.Threading
open Microsoft.AspNet.SignalR
open Microsoft.AspNet.SignalR.Hubs
open System.Reactive.Subjects
open System.Reactive.Linq
open StockTickerServer
open StockTickerHubClient

// speclized agent for each user
// keep track orders and status of portfolio
type TradingAgent(id : string, initialAmount : float, caller:IHubCallerConnectionContext<IStockTickerHubClient>) =
        
    let connId = id
    let caller = caller
      
    let agent =
        new Agent<TradingMessage>(fun inbox ->
            // single thread safe no sharing
            let rec loop cash (portfolio : Portfolio) (buyOrders : Treads) (sellOrders : Treads) =
                async {
                    let! msg = inbox.Receive()
                    match msg with
                    | Kill(reply) -> reply.Reply()
                                                
                    | Error(exn) -> raise exn
                      
                    | TradingMessage.AddStock(s) ->
                        caller.Client(connId).SetStock(s)
                        return! loop cash portfolio buyOrders sellOrders                    
                      
                    | TradingMessage.Buy(symbol, trading) ->
                        let items = setOrder buyOrders symbol trading // let see the Treads type
                        let buyOrder = createOrder symbol trading "Buy"
                        caller.Client(connId).UpdateOrderBuy(buyOrder)
                        return! loop cash portfolio items sellOrders
                      
                    | TradingMessage.Sell(symbol, trading) ->
                        let items = setOrder sellOrders symbol trading 
                        let sellOrder = createOrder symbol trading "Sell"
                        caller.Client(connId).UpdateOrderSell(sellOrder)
                        return! loop cash portfolio buyOrders items
                      
                    | TradingMessage.UpdateStock(stock) ->
                            caller.Client(connId).UpdateStockPrice stock

                            let symbol = stock.Symbol
                            let price = stock.Price
                           
                            let updatedPortfolioBySell = 
                                updatePortfolioBySell symbol (portfolio:Portfolio) (sellOrders:Treads) price 
                            
                            let cashAfterSell, portfolio', sellOrders' = 
                                match updatedPortfolioBySell with
                                | None -> cash, portfolio, sellOrders
                                | Some(r, p, s) -> (cash + r), p, s

                            let updatedPortfolioByBuy = 
                                updatePortfolioByBuy symbol portfolio' buyOrders cashAfterSell price 
                            
                            let cashAfterBuy, portfolio'', buyOrders' =
                                match updatedPortfolioByBuy with
                                | None -> cashAfterSell, portfolio', buyOrders
                                | Some(c, p, b) -> (cash - c), p, b
  
                            let asset = getUpdatedAsset portfolio'' sellOrders' buyOrders' cashAfterBuy 
                            
                            caller.Client(connId).UpdateAsset(asset)
                            
                            return! loop cashAfterBuy portfolio buyOrders sellOrders
                }
            loop initialAmount (Portfolio(HashIdentity.Structural)) (Treads(HashIdentity.Structural))
                (Treads(HashIdentity.Structural)))
 
    member x.Agent = agent
    member x.Id = connId
 
    // TradingAgent implement the IObserver interface
    interface IObserver<TradingMessage> with
        member x.OnNext(msg) = agent.Post(msg:TradingMessage)
        member x.OnError(exn) = agent.Post(Error exn)
        member x.OnCompleted() = agent.PostAndReply(Kill)