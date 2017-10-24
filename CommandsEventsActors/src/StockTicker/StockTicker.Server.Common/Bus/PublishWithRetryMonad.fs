namespace StockTicker

open System
open StockTicker.Events
open StockTicker.Core
open EventStorage
open StockMarket
open Events


type SendMessageWith<'a> =
    | SendMessageWith of string * 'a


    // Computation expressions in F# provide a convenient syntax for writing
    // computations that can be sequenced and combined using control flow constructs
    // and bindings. They can be used to provide a convenient syntax for monads,
    // a functional programming feature that can be used to manage data, control,
    // and side effects in functional programs.

    // Computation expressions in F# provide a convenient syntax for writing computations
    // that can be sequenced and combined using control flow constructs and bindings.

[<AutoOpenAttribute>]
module RetryPublishMonad =
    let stockMarket = SingletonStockMarket.InstanceStockMarket()

    let Storage = new EventStorage()


    type SendCommand<'a> =
        | SendCommandWith of string * 'a


    type RetryPublishBuilder(max, sleepMilliseconds : int) =
        member x.Return(a) = a
        member x.Bind(SendCommandWith(connId, commandWrapper:CommandWrapper) : SendCommand<_>, fn) =

            let rec loop n (error:exn option) =
                async {
                    if n = 0 then

                        let cmd = commandWrapper.Command
                        match cmd with
                        | BuyStockCommand(connId,trading)
                        | SellStockCommand(connId, trading) ->
                                    // notify the error
                                    // and persiste the event into the Event-Storage
                                    let eventExn = ErrorSubmitingOrder(Guid(connId), trading, error.Value)
                                    (Guid(connId), Event.CreateEventDescriptor(Guid(connId), eventExn))
                                    ||> Storage.SaveEvent
                                    failwith "Failed Theading"

                        | AddStockCommand(connId, ticker) -> failwith "Failed Adding Ticker"


                    else
                        try

                            // publish the command
                            stockMarket.PublishCommand(connId, commandWrapper)

                            let event =
                                        let cmd = commandWrapper.Command
                                        match cmd with
                                        | BuyStockCommand(connId,trading)  ->  StocksBuyedEvent(commandWrapper.Id, trading)
                                        | SellStockCommand(connId, trading) -> StocksSoldEvent(commandWrapper.Id, trading)                       | AddStockCommand(connId, ticker) -> AddedTicker(commandWrapper.Id, ticker)

                            // and persiste the event into the Event-Storage
                            (Guid(connId), Event.CreateEventDescriptor(commandWrapper.Id, event))
                            ||> Storage.SaveEvent

                        with ex ->
                            sprintf "Call failed with %s. Retrying." ex.Message |> printfn "%s"
                            do! Async.Sleep sleepMilliseconds
                            return! loop (n - 1) (Some(ex))
                }
            loop max None |> Async.Start

    let retryPublish = RetryPublishBuilder(3, 1000)



