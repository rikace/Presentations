module Server

open Akka.FSharp
open Akka.Actor
open Akka.Remote
open Akka.Configuration
open System
open System.IO

let config =
    Configuration.parse
        @"akka {
            actor.provider = ""Akka.Remote.RemoteActorRefProvider, Akka.Remote""
            remote.helios.tcp {
                hostname = localhost
                port = 9238
            }
        }"

[<EntryPoint>]
let main argv = 
    use localSystem = System.create "local-system" config

    // (4)  repeat point (3) 
    //      use location transperency to localize the address of the remote Actor created in the Client project,
    //      then, with the Client project running, send a message to the remote Actor
    //      Note: The actor system running in the Client project has a different name than the local one
    let remoteActorSel =  " < CODE HERE > "
    // remoteActorSel <! "Hello from here!"

    // (5)  create an Actor using code quotation <@@>,
    //      then remote deploy the Actor created to the Client project
    //      use the function "spawnRemote"
    //      - try different implementation
    
    //  Remote deployment in Akka F# is done through spawne function 
    //  and it requires deployed code to be wrapped into F# quotation.

    //  This is an helper function to deploy an actor remotely
    //  The logic informs the local system that deployment should occur on the remote machine. 
    //  The full address must be provided, including the network localization and protocol type used for communication
    //  The option SpawnOption.Deploy specifies what deployment is meant to occur. 
    let spawnRemote systemOrContext remoteSystemAddress actorName expr =
        spawne systemOrContext actorName expr [SpawnOption.Deploy (Deploy(RemoteScope (Address.Parse remoteSystemAddress)))]
    
    //  the Remote deployment in Akka F# is done through the spawne function
    //  and it requires the deployed code to be wrapped into F# quotation.

    //  With F# <@@> remote-deployment we don’t define actors in shared libraries, which have to be bound to both endpoints. 
    //  Actor logic is compiled in the runtime, while remote actor system is operating. 
    //  That means, there is no need to stop your remote nodes to reload shared actor assemblies when updated. 
    //  Note: Code embedded inside quotation must use only functions

    let aref = " < CODE HERE > "
        // spawnRemote 

    aref <! "Message Here"                                   
    
    Console.ReadLine() |> ignore    
    localSystem.Terminate().Wait()
    0


