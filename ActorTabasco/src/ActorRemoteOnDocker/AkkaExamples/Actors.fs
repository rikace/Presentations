namespace AkkaExamples

open System
open Akka.Actor
open Akka.Configuration
open Akka.FSharp
    

module ActorsMoudle =
    type EchoServer =
        inherit Actor

        override x.OnReceive message =
            match message with
            | :? string as msg -> printfn "Hello %s" msg
            | _ ->  failwith "unknown message"
