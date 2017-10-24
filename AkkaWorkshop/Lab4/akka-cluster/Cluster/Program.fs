open Akka.FSharp
open Akka.FSharp.Spawn
open Akka.Actor
open FSharp.Configuration
open System
open WriterActor
open ReaderActor

[<EntryPoint>]
let main argv =
     // (1)  update the config file to run this Actor-System as node of a cluster 

    printfn "Staring..."

    let system = System.create "myActorSystem" (Configuration.load())

    let routerOpt = SpawnOption.Router ( Akka.Routing.FromConfig.Instance )
    let supervisionOpt = SpawnOption.SupervisorStrategy (Strategy.OneForOne(fun _ -> Directive.Stop))

    let writer = spawn system "WriterActor" (WriterActor)

    let reader = spawne system "ReaderActor" <@ (ReaderActor) @> [routerOpt; supervisionOpt]

    system.Scheduler.ScheduleTellRepeatedly(TimeSpan.FromSeconds(1.), TimeSpan.FromSeconds(1.), reader, ReadMessage)
   
    printfn "Listening..."
    system.WhenTerminated.Wait()

    printfn "Step End"
    0