module RemoteDeploy


#if INTERACTIVE
#r "../Akka.dll"
#r "../Akka.FSharp.dll"
#r "../Akka.Remote.dll"
#r "../System.Collections.Immutable.dll"
#endif

open Akka.FSharp
open Akka.Actor
open Akka.Remote
open Akka.Configuration
open System
open System.IO

//System.Diagnostics.Process.Start(@"C:\Git\AkkaActorModel\RemoteAkka.Console\bin\Debug\RemoteAkka.Console.exe") |> ignore

let config =
    Configuration.parse
        @"akka {
            actor.provider = ""Akka.Remote.RemoteActorRefProvider, Akka.Remote""
            remote.helios.tcp {
                hostname = localhost
                port = 9234
            }
        }"


// return Deploy instance able to operate in remote scope
let deployRemotely address = Deploy(RemoteScope (Address.Parse address))

// Remote deployment in Akka F# is done through spawne function and it requires deployed code to be wrapped into F# quotation.
let spawnRemote systemOrContext remoteSystemAddress actorName expr =
 spawne systemOrContext actorName expr [SpawnOption.Deploy (deployRemotely remoteSystemAddress)]

 
let localSystem = System.create "local-system" config

let aref =
    spawnRemote localSystem "akka.tcp://remote-system@192.168.99.100:9234/" "hello"
  //  spawnRemote localSystem "akka.tcp://remote-system@172.17.0.2:9234/" "hello" 
    // spawnRemote localSystem "akka.tcp://remote-system@10.211.55.2:9234/" "hello" 
    // actorOf wraps custom handling function with message receiver logic

      <@ actorOf (fun msg -> System.Console.ForegroundColor <- System.ConsoleColor.Red
                             printfn "received 10  '%s'" msg) @>
                                   

let arefCtx =
//    spawnRemote localSystem "akka.tcp://remote-system@192.168.99.102:9234/" "hello"
    spawnRemote localSystem "akka.tcp://remote-system@192.168.1.10:9234/" "hello" 
   // spawnRemote localSystem "akka.tcp://remote-system@10.211.55.2:9234/" "hello" 
    // actorOf wraps custom handling function with message receiver logic

//      <@ actorOf (fun msg -> System.Console.ForegroundColor <- System.ConsoleColor.Red
//                             printfn "received 10  '%s'" msg) @>

      <@ actorOf2 (fun ctx msg -> printfn "%A received: %s" ctx.Self msg) @>



// send example message to remotely deployed actor
aref <! "Hello NYC F# UG!! from F# and AKKA.Remote"

// thanks to location transparency, we can select 
// remote actors as if they where existing on local node
let sref = select "akka://local-system/user/hello" localSystem
sref <! "Hello again"



// we can still create actors in local system context
let lref = spawn localSystem "local" (actorOf (fun msg -> printfn "local '%s'" msg))
// this message should be printed in local application console
lref <! "Hello locally"


    