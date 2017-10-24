
open Akka.FSharp
open Akka.Actor
open Akka.Remote
open Akka.Configuration
open System
open System.Threading.Tasks

module ActorsMoudle =
    type EchoServer =
        inherit Actor

        override x.OnReceive message =
            match message with
            | :? string as msg -> printfn "Hello %s" msg
            | _ ->  failwith "unknown message"

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
                port = 8091              
                hostname = "10.211.55.2" #0.0.0.0 
                }
        }
        """

    System.Console.Title <- "Remote: " + System.Diagnostics.Process.GetCurrentProcess().Id.ToString()

    use remoteSystem = System.create "remote-system" (Configuration.parse config)

    Console.ForegroundColor <- ConsoleColor.Green
    printfn "Remote Actor %s listening..." remoteSystem.Name

    let echoActor = spawn remoteSystem "echoActor" (actorOf2 (fun mailbox m -> printfn "%A said %s" (mailbox.Self.Path) m))
    
    System.Console.ReadLine() |> ignore
    remoteSystem.Terminate().Wait()
    0

    // (1)  create Docker image pushing this project.
    //      run the docker image and send messageses
    //      to send messages to the docker image you can use F# Interactive            