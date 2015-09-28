namespace StockTicker.Commands

open System
open System.Collections.Generic
open StockTicker.Events
open EventStorage
open StockTicker.Core
open System.Threading.Tasks
open StockTicker
open Command
open Events



// The verbs (actions) of the system (in imperative mood / present tense)
module CommandHandler =

    type ICommand = interface end

    type IHandleCommand<'T when 'T :> ICommand> = 
        abstract Handle : 'T -> Action -> Task

    let Storage = new EventStorage() 

    let AsyncHandle (msg:TradingCommand) = 
        async {
            let action = 
                //let fetchitem id = new InventoryItem(id) |> Storage.GetHistoryById id
                //let treadingActor = SingletonHub.SingletonHub.InstanceTreadingActor()

                match msg with
                | BuyStock(connId, treading) ->
                    retryPublish {
                        let order = {ticker= treading.Ticker; quantity=treading.Quantity; price=treading.Price; oderType=OrderType.Buy}
                     
                        do! SendCommandWith( connId, order)
                    }                     

//                    let commandDescriptor = CommandWrapper.CreateMessageCommand connId treading
//                    let event = StocksBuyedEvent(commandDescriptor.Id, treading)
//                    (commandDescriptor.Id, Event.CreateEventDescriptor(commandDescriptor.Id, event))  ||> Storage.SaveEvent 

                     // treadingActor.PostBuyOrder(connId, treading.Ticker, {Quantity = treading.Quantity; Price = treading.Price})
                       // publish to buy and sel servuce

                | SellStock(connId, treading) -> 
                    retryPublish {
                        let order = {ticker= treading.Ticker; quantity=treading.Quantity; price=treading.Price; oderType=OrderType.Sell}
                     
                        do! SendCommandWith( connId, order)
                    }                     
                    //treadingActor.PostSellOrder(connId, treading.Ticker, {Quantity = treading.Quantity; Price = treading.Price})
                       // publish
//
//                    let commandDescriptor = CommandWrapper.CreateMessageCommand connId treading
//                    let event = StocksSoldEvent(commandDescriptor.Id, treading)
//                    (commandDescriptor.Id, Event.CreateEventDescriptor(commandDescriptor.Id, event))  ||> Storage.SaveEvent 
                       
                | TikcerRequest(connId, tickerRequest) -> 
                    retryAddTicker {
                        do! SendCommandWith(connId, tickerRequest)
                    }
                                            
            action |> ignore
        }
        
    let Handle (msg:TradingCommand) =  
                let handleResponse = AsyncHandle msg |> Async.Catch |> Async.RunSynchronously
                match handleResponse with 
                | Choice1Of2(_) -> ()
                | Choice2Of2(exn) -> ()

    
    
    //.RunSynchronously



//(Storage :?> EventStorage).ShowItemHistory itemId1
//let storedItemName = InMemoryDatabase.InventoryItems.[0].Name
//let storedDetailName = InMemoryDatabase.InventoryItemDetails.[itemId1].Name