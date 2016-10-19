//namespace AkkaExamples

open Akka.FSharp
open Akka.Actor
open Akka.Remote
open Akka.Configuration
open System
open System.Threading.Tasks

type EchoServer =
    inherit UntypedActor

    override x.OnReceive message =
        match message with
        | :? string as msg -> printfn "Hello %s" msg
        | _ ->  failwith "unknown message"

[<EntryPoint>]
let main argv = 
(*    let config = """
        akka {  
            akka.suppress-json-serializer-warning
            actor {
                provider = "Akka.Remote.RemoteActorRefProvider, Akka.Remote"
            }    
            remote.helios.tcp {
                transport-protocol = tcp
                port = 9234                 
                hostname = 10.211.55.2
               
            }
        }
        """
*)
// akka.suppress-json-serializer-warning

// docker run -i --name test1 -p 9234:9234 rikace/akka1
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
                hostname =  0.0.0.0
                }
        }
        """

    System.Console.Title <- "Remote: " + System.Diagnostics.Process.GetCurrentProcess().Id.ToString()

    use remoteSystem = System.create "remote-system" (Configuration.parse config)
    Console.ForegroundColor <- ConsoleColor.Green

    printfn "Remote Actor %s listening..." remoteSystem.Name
 
    System.Console.ReadLine() |> ignore

    remoteSystem.Terminate().Wait()


    //let system = ActorSystem.Create("FSharp")

(*    let echoServer = 
        spawn system "EchoServer"
        <| fun mailbox ->
                actor {
                    let! message = mailbox.Receive()
                    match box message with
                    | :? string as msg -> printfn "Hello %s" msg
                    | _ ->  failwith "unknown message"
            } 
    echoServer <! "F#!"
    *)
    Console.ReadLine() |> ignore

  //  system.Shutdown()
    0 // return an integer exit code

