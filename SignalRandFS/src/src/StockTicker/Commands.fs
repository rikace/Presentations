namespace StockTicker.Commands

open System
open System
open StockTicker.Rop
open System.Threading.Tasks

[<AutoOpen>]
module Command = 
    type TradingRecord = 
        { Ticker : string
          Quantity : int
          Price : float
          Trading : Trading }
    
    and Trading = 
        | Buy
        | Sell
    
    type TradingCommand = 
        | BuyStock of connectionId : string * tradingRecord : TradingRecord
        | SellStock of connectionId : string * tradingRecord : TradingRecord
        | TikcerRequest of connectionId : string * TickerRequestRecord
    
    and TickerRequestRecord = 
        { Ticker : string
          Price : float }
    
    [<CLIMutable>]
    type CommandWrapper = 
        { ConnectionId : string
          Id : Guid
          Created : DateTimeOffset
          Command : TradingCommand }
        
        static member CreateTreadingCommand connectionId (item : TradingRecord) = 
            let command = 
                match item.Trading with
                | Buy -> BuyStock(connectionId, item)
                | Sell -> SellStock(connectionId, item)
            { Id = (Guid.NewGuid())
              Created = (DateTimeOffset.Now)
              ConnectionId = connectionId
              Command = command }
        
        static member CreateTickerRequestCommand connectionId (item : TickerRequestRecord) = 
            let command = TikcerRequest(connectionId, item)
            { Id = (Guid.NewGuid())
              Created = (DateTimeOffset.Now)
              ConnectionId = connectionId
              Command = command }
    
    let validateTicker (input : TradingRecord) = 
        if input.Ticker = "" then Failure "Ticket must not be blank"
        else Success input
    
    let validateQuantity (input : TradingRecord) = 
        if input.Quantity <= 0 || input.Quantity > 50 then Failure "Quantity must be positive and not be more than 50"
        else Success input
    
    let validatePrice (input : TradingRecord) = 
        if input.Price <= 0. then Failure "Price must be positive"
        else Success input
    
    let tradingdValidation = validateTicker &&& validateQuantity &&& validatePrice
    
    let validateTickerRequestSymbol (input : TickerRequestRecord) = 
        if input.Ticker = "" then Failure "Ticket must not be blank"
        else Success input

    let validateTickerRequestPriceMin (input : TickerRequestRecord) = 
        if input.Price <= 0. then Failure "Price must be positive"
        else Success input
    
    let validateTickerRequestPriceMax (input : TickerRequestRecord) = 
        if input.Price >= 1000. then Failure "Price must be positive"
        else Success input
    
    let tickerRequestValidation = 
        validateTickerRequestSymbol &&& validateTickerRequestPriceMin &&& validateTickerRequestPriceMax
//r|> validateTickerRequestSymbol >>= validateTickerRequestPrice
//        let valPrice = bind validateTickerRequestPrice        
//        validateTickerRequestSymbol >> valPrice
