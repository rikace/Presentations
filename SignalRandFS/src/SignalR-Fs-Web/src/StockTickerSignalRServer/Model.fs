namespace StockTickerServer

open System
open System.Collections.Generic
open Microsoft.AspNet.SignalR
open Microsoft.AspNet.SignalR.Hubs

[<AutoOpenAttribute>]
module Models = 

    [<CLIMutable>] 
    // using the CLIMutable attribute the Record type 
    // is serializable by other .NET langauges/frameworks
    // while the type is still immutable by F#
    type OrderRecord = 
        { Symbol : string
          Quantity : int
          Price : float
          OrderType : string }

    [<CLIMutable>]
    type Asset = {Cash:float; Portfolio:List<OrderRecord>; BuyOrders:List<OrderRecord>; SellOrders:List<OrderRecord>}

    [<CLIMutable>]
    type TradingRequest = 
        { ConnectionID : string
          Symbol : string
          Price : float
          Quantity : int }

    [<CLIMutable>]
    type TradingRecord = 
        { Symbol : string
          Quantity : int
          Price : float
          Trading : TradingType }    
    and TradingType = 
        | Buy
        | Sell
    
    type TradingCommand = 
        | BuyStockCommand of connectionId : string * tradingRecord : TradingRecord
        | SellStockCommand of connectionId : string * tradingRecord : TradingRecord
        | AddStockCommand of connectionId : string * TickerRecord       
   
    and [<CLIMutableAttribute>]
        TickerRecord = 
        { Symbol : string
          Price : float }
    
    [<CLIMutable>]
    type CommandWrapper = 
        { ConnectionId : string
          Id : Guid
          Created : DateTimeOffset
          Command : TradingCommand }
        
        static member CreateTradingCommand connectionId (item : TradingRecord) = 
            let command = 
                match item.Trading with
                | Buy -> BuyStockCommand(connectionId, item)
                | Sell -> SellStockCommand(connectionId, item)
            { Id = (Guid.NewGuid())
              Created = (DateTimeOffset.Now)
              ConnectionId = connectionId
              Command = command }
        
        static member CreateTickerRequestCommand connectionId (item : TickerRecord) = 
            let command = AddStockCommand(connectionId, item)
            { Id = (Guid.NewGuid())
              Created = (DateTimeOffset.Now)
              ConnectionId = connectionId
              Command = command }
    
    [<CLIMutableAttribute>]
    type Stock = 
        { Symbol : string
          DayOpen : float
          DayLow : float
          DayHigh : float
          LastChange : float
          Price : float
          Index : int }
        member x.Change = x.Price - x.DayOpen
     
        member x.PercentChange = double (Math.Round(x.Change / x.Price, 4))        
     
        static member CreateStock (symbol : string) price index = 
            { Symbol = symbol
              LastChange = 0.
              Price = price
              DayOpen = 0.
              DayLow = 0.
              DayHigh = 0.
              Index = index }        
     
        static member InitialStocks() = 
            seq { 
                for stock in [ ("MSFT", 41.68, 0)
                               ("APPL", 92.08, 1)
                               ("AMZN", 380.15, 2)
                               ("GOOG", 543.01, 3)
                               ("FB", 78.97, 4)] -> stock |||> Stock.CreateStock }
    
    type MarketState = 
        | Closed
        | Open    

    type StockTickerMessage = 
        | UpdateStock of Stock
        | UpdateStockPrices 
        | PublishCommand of connId : string * CommandWrapper
        | OpenMarket of string
        | CloseMarket of string
        | GetMarketState of string * AsyncReplyChannel<MarketState>
        | GetAllStocks of string * AsyncReplyChannel<Stock list>


    type TradingMessage =
        | Kill of AsyncReplyChannel<unit>
        | Error of exn
        | Buy of symbol : string * TradingDetails 
        | Sell of symbol : string * TradingDetails
        | UpdateStock of Stock
        | AddStock of Stock   
    
    and TradingDetails =
        { Quantity : int
          Price : float
          TradingType:string }
   
    and Treads = Dictionary<string, ResizeArray<TradingDetails>>
   
    and Portfolio = Dictionary<string, TradingDetails>                                                                                 