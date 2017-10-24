module WriterActor

open System
open Akka.Actor
open Akka.FSharp

type WriterMessage =
    | WriteMessage of string

let WriterActor (mailbox: Actor<WriterMessage>) =

    printfn "Writer actor is listening..."

    let rec reader() = actor {
        let! msg = mailbox.Receive()

        match msg with
        | WriteMessage s -> printfn "Write %s" s

        return! reader()
        }

    reader()

