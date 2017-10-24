// Learn more about F# at http://fsharp.org
// See the 'F# Tutorial' project for more help.

open Akka.FSharp
open Akka.FSharp.Spawn
open Akka.Actor

[<EntryPoint>]
let main argv = 
    // (1)  update the config file to run this Actor-System as node of a cluster 

    let system = System.create "myActorSystem" (Configuration.load())
    system.WhenTerminated.Wait()
    0 // return an integer exit code
