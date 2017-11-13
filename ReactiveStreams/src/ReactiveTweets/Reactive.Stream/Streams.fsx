
#r "../src/Akka.dll"
#r "../src/Hyperion.dll"
#r "../src/Newtonsoft.Json.dll"
#r "../src/FSharp.PowerPack.dll"
#r "../src/FSharp.PowerPack.Linq.dll"
#r "../src/Akkling.dll"
#r "../src/Reactive.Streams.dll"
#r "../src/Akka.Streams.dll"
#r "../src/Akkling.Streams.dll"
#r "../src/System.Collections.Immutable.dll"

open System
open Akka.Streams
open Akka.Streams.Dsl
open Akkling
open Akkling.Streams
open Akkling.Behaviors

let text = """
       Lorem Ipsum is simply dummy text of the printing and typesetting industry.
       Lorem Ipsum has been the industry's standard dummy text ever since the 1500s,
       when an unknown printer took a galley of type and scrambled it to make a type
       specimen book."""

let system = System.create "streams-sys" <| Configuration.defaultConfig()
let mat = system.Materializer()

let processText source =
    source
    |> Source.map (fun (x:string) -> x.ToUpper())
    |> Source.filter (String.IsNullOrWhiteSpace >> not)
    |> Source.runForEach mat (printfn "%s")


Source.ofArray (text.Split()) |> processText |> Async.Start

// val behavior : targetRef:ICanTell<'a> -> m:Actor<'a> -> Effect<'a>
let behavior targetRef (m:Actor<_>) =
    let rec loop () = actor {
        let! msg = m.Receive ()
        targetRef <! msg
        return! loop ()
    }
    loop ()

// val spawnActor : targetRef:ICanTell<'a> -> IActorRef<'a>
let spawnActor targetRef =
    spawnAnonymous system <| props (behavior targetRef)


let echo = Source.actorRef OverflowStrategy.DropNew 1000
        |> Source.mapMaterializedValue(spawnActor)
        |> Source.toMat(Sink.forEach(fun s -> printfn "Received: %s" s)) Keep.left
        |> Graph.run mat

echo <! "Hello!!"


let processTextCombined =
        Source.actorRef OverflowStrategy.DropNew 1000
        |> Source.mapMaterializedValue(spawnActor)
        |> Source.collect(fun (text:string) -> text.Split())
        |> Source.filter (String.IsNullOrWhiteSpace >> not)
        |> Source.map (fun x -> x.ToUpper())
        |> Source.toMat(Sink.forEach(fun s -> printfn "Received: %s" s)) Keep.left
        |> Graph.run mat

processTextCombined <! text
