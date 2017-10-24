module ReaderActor

open System
open Akka.Actor
open Akka.FSharp

open WriterActor

type ReaderMessage =
    | ReadMessage

let ReaderActor (mailbox: Actor<ReaderMessage>) =

    printfn "Reader actor is listening..."

    let rnd = new Random()

    let rec reader() = actor {
            let! msg = mailbox.Receive()

            match msg with
            | ReadMessage ->
                printfn "Read : message received"
                let writer = mailbox.ActorSelection("../../WriterActor")
                writer <! WriteMessage(rnd.Next().ToString())

            return! reader()
        }

    reader()

