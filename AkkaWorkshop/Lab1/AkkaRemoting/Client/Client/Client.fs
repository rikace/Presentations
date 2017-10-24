module Client

open Akka.FSharp
open Akka.Actor
open Akka.Remote
open Akka.Configuration
open System
open System.Threading.Tasks

[<EntryPoint>]
let main argv = 
    let config = """
        akka {  
            log-config-on-start = on        
            stdout-loglevel = DEBUG
            loglevel = DEBUG
          
            actor {
                provider = "Akka.Remote.RemoteActorRefProvider, Akka.Remote"
            }    
            remote.helios.tcp {
                transport-protocol = tcp
                port = 9234                 
                hostname = localhost  
                }
        }
        """
    System.Console.Title <- "Client : " + System.Diagnostics.Process.GetCurrentProcess().Id.ToString()

    use remoteSystem = System.create "remote-system" (Configuration.parse config)

    Console.ForegroundColor <- ConsoleColor.Green
    printfn "Remote Actor %s listening..." remoteSystem.Name

    // (1)  create and start an Actor using the spawn function.
    //      Explore functions to define an Akka Actor such as actorOf2 & actorOf
    //      In this case, the Actor can be simple as receiving a string as message type and printing  
    //      into the console some response... maybe with different text color ;) 

    let aref = "...actor here ..."

    // (2)  send a message to the Actor just implemented (1)
    //      you can try to send a message (Tell) or send-and-receive a reponse (Ask)
    
    // < CODE HERE >

    // (3)  use location transperency to localize the address of the Actor created (1),
    //      then, using this address send a message.
    //      The F# API has a function "select" to ActorSelection 

    let actorSel = " < CODE HERE > "
    // actorSel <! "Hello once more"

    System.Console.ReadLine() |> ignore

    remoteSystem.Terminate().Wait()

    Console.ReadLine() |> ignore
    0



