namespace StockTicker.Events

open System
open StockTicker.Commands
open System.Threading.Tasks
open System.Collections.Generic


// The verbs of the system (in imperfect form)
module Events =

    // Events implemented as discriminated union. 
    // If you use a big solution, change to a base type 
    // or just use many event storages and concatenate / merge them with LINQ 
    type Event =
        | StocksBuyedEvent of Guid * TradingRecord
        | StocksSoldEvent of Guid * TradingRecord
        | ErrorSubmitingOrder of Guid * TradingRecord * exn
        override x.ToString() = 
            match x with
            | StocksBuyedEvent(id, trading) -> 
                    sprintf "Item Id %A - Ticker %s sold at $ %f - quantity %d" id trading.Ticker trading.Price trading.Quantity
            | StocksSoldEvent(id, trading) -> 
                    sprintf "Item Id %A - Ticker %s buyed at $ %f - quantity %d" id trading.Ticker trading.Price trading.Quantity 
        
        static member CreateEventDescriptor (id:Guid, eventData:Event) = 
                EventDescriptor(id, eventData)

    // Container to capsulate events
    and EventDescriptor(id:Guid, eventData:Event) = 
        member x.Id = id
        member x.EventData = eventData

