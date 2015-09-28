namespace StockTicker

open System
open StockTicker.Commands
open StockTicker.Events
open StockTicker.Core
open EventStorage
open Command
open Events

type SendMessageWith<'a> = 
    | SendMessageWith of string * 'a

[<AutoOpenAttribute>]
module RetryPublishMonad = 
    let stockTicker = SingletonHub.SingletonHub.InstanceStockTicker()
    
    let Storage = new EventStorage() 


    type SendCommandWith<'a> = 
        | SendCommandWith of string * 'a
    
    type RetryPublishBuilder(max, sleepMilliseconds : int) = 
        member x.Return(a) = a
        member x.Bind(SendCommandWith(connId, msg) : SendCommandWith<_>, fn) = 
            let treadingType =  function 
                                | OrderType.Buy -> Trading.Buy
                                | OrderType.Sell -> Trading.Sell
                            
            let treading = 
                {   Ticker = msg.ticker
                    Quantity = msg.quantity
                    Price = msg.price
                    Trading = treadingType msg.oderType }

            let rec loop n (error:exn option) = 
                async { 
                    if n = 0 then 
                        let eventExn = ErrorSubmitingOrder(Guid(connId), treading, error.Value) 
                        (Guid(connId), Event.CreateEventDescriptor(Guid(connId), eventExn))
                        ||> Storage.SaveEvent
                        failwith "Failed"

                    else 
                        try 
                            stockTicker.PublishOrder(connId, msg)
                            
                            let commandDescriptor = CommandWrapper.CreateTreadingCommand connId treading
                            let event = StocksSoldEvent(commandDescriptor.Id, treading)
                            
                            (Guid(connId), Event.CreateEventDescriptor(commandDescriptor.Id, event)) 
                            ||> Storage.SaveEvent

                        with ex ->                             
                            sprintf "Call failed with %s. Retrying." ex.Message |> printfn "%s"
                            do! Async.Sleep sleepMilliseconds
                            return! loop (n - 1) (Some(ex))
                }
            loop max None |> Async.Start


    type RetryAddTickerBuilder(max, sleepMilliseconds : int) = 
        member x.Return(a) = a
        member x.Bind(SendCommandWith(connId, msg:TickerRequestRecord) : SendCommandWith<_>, fn) = 
        
            let rec loop n (error:exn option) = 
                async { 
                    if n = 0 then                         
                        failwith "Failed"
                    else 
                        try 
                            stockTicker.AddStock msg.Ticker msg.Price 

                        with ex ->                             
                            sprintf "Call failed with %s. Retrying." ex.Message |> printfn "%s"
                            do! Async.Sleep sleepMilliseconds
                            return! loop (n - 1) (Some(ex))
                }
            loop max None |> Async.Start

    
    let retryPublish = RetryPublishBuilder(3, 1000)
    let retryAddTicker = RetryAddTickerBuilder(3, 1000)
