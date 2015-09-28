namespace StockTicker.Commands

open System
open System.Collections.Generic
open StockTicker.Events
open EventStorage
open StockTickerServer
open System.Threading.Tasks
open StockTicker
open Events



// The verbs (actions) of the system (in imperative mood / present tense)
module CommandHandler =
 
    let Storage = new EventStorage() 

    let AsyncHandle (cmd:CommandWrapper) = 
        async { // async workflow
       
            let { ConnectionId = connectionId
                  Id = id  
                  Created = created
                  Command = tradingCommand } = cmd
            
            retryPublish {
                do! SendCommandWith( connectionId, cmd)
            }                                 
        }

        
    let Handle (msg:CommandWrapper) =  
                let handleResponse = AsyncHandle msg |> Async.Catch |> Async.RunSynchronously
                match handleResponse with 
                | Choice1Of2(_) -> ()
                | Choice2Of2(exn) -> ()



//(Storage :?> EventStorage).ShowItemHistory itemId1
//let storedItemName = InMemoryDatabase.InventoryItems.[0].Name
//let storedDetailName = InMemoryDatabase.InventoryItemDetails.[itemId1].Name