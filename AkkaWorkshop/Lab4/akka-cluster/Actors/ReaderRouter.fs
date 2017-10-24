module ReaderRouter

open System
open Akka.Actor
open Akka.FSharp

open ReaderActor

type ReaderRouterMessage = 
    | ReaderRouterStart

let ReaderRouter (writer: IActorRef) (mailbox: Actor<ReaderRouterMessage>) = 
    
    let routerOpt = SpawnOption.Router ( Akka.Routing.FromConfig.Instance )
    let supervisionOpt = SpawnOption.SupervisorStrategy (Strategy.OneForOne(fun _ -> Directive.Stop))

    let rec router() = 
        actor {
            
            let! msg = mailbox.Receive()

            match msg with 
            | ReaderRouterStart ->
                let reader = spawnOpt mailbox "ReaderActor" ReaderActor [routerOpt; supervisionOpt] 
                return! router()                                  
        }
        
    router()