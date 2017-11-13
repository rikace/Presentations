module Reactive.StreamOne

open System
open Akka
open Akka.Actor
open Akka.Streams
open Akka.Streams.Dsl
open Tweetinvi.Models
open Shared.Reactive
open System.Runtime.CompilerServices
open System.Collections.Generic
open System.Linq
open Akka.Streams
open System.Xml.Linq

let inline select map (flow:Flow<_,_, NotUsed>) =
    flow.Select(new Func<_,_>(map))

let inline where (predicate:Predicate<_>) (source:Source<_,_>) = source.Where(predicate)

let inline throttle (elements:int) (per:TimeSpan) (maximumBurst:int) (mode:ThrottleMode) (flow:Flow<_,_,_>) =
    flow.Throttle(elements, per, maximumBurst, mode)

// Statically resolved parameters
let inline via< ^a, ^b, ^c when ^a : (member Via: ^b -> ^c )> (b: ^b) a =
      (^a : (member Via: ^b -> ^c) (a, b))

let inline sink< ^a, ^b, ^c when ^a : (member To: ^b -> ^c )> (b: ^b) a =
      (^a : (member To: ^b -> ^c) (a, b))

let inline from< ^a, ^b, ^c when ^a : (member From: ^b -> ^c )> (b: ^b) a =
      (^a : (member From: ^b -> ^c) (a, b))

module TweetsToConsole =

    let inline create<'a>(tweetSource :Source<ITweet, 'a>) : IRunnableGraph<'a> =
        let formatFlow =
            Flow.Create<ITweet>()
            |> select (Utils.FormatTweet)

        let writeSink = Sink.ForEach<string>(fun text -> Console.WriteLine(text))

        tweetSource
        |> via formatFlow
        |> sink writeSink

module TweetsWithBroadcast =

   let inline create(tweetSource:Source<ITweet, 'a>) =
        let formatUser =
            Flow.Create<IUser>()
            |> select (Utils.FormatUser)
        let formatCoordinates =
            Flow.Create<ICoordinates>()
            |> select (Utils.FormatCoordinates)
        let flowCreateBy =
            Flow.Create<ITweet>()
            |> select (fun tweet -> tweet.CreatedBy)
        let flowCoordinates =
            Flow.Create<ITweet>()
            |> select (fun tweet -> tweet.Coordinates)

        let writeSink = Sink.ForEach<string>(fun text -> Console.WriteLine(text))
        let via (b:IGraph<FlowShape<_, _>, NotUsed>) (a:GraphDsl.ForwardOps<_, NotUsed>) = a.Via(b)
        let via' (b:IGraph<FlowShape<_, _>, NotUsed>) (a:Source<ITweet, _>) = a.Via(b)
        let to' (b:Inlet<_>) (a:GraphDsl.ForwardOps<_, NotUsed>) = a.To(b)

        let graph = GraphDsl.Create(fun buildBlock ->
            let broadcast = buildBlock.Add(Broadcast<ITweet>(2))
            let merge = buildBlock.Add(Merge<string>(2))

            buildBlock
            |> from (broadcast.Out(0))
            |> via flowCreateBy
            |> via formatUser
            |> to' (merge.In(0))
            |> ignore

            buildBlock
            |> from (broadcast.Out(1))
            |> via flowCoordinates
            |> via formatCoordinates
            |> to' (merge.In(1))
            |> ignore

            FlowShape<ITweet, string>(broadcast.In, merge.Out))

        tweetSource
        |> where (Predicate(fun (tweet:ITweet) -> not(isNull tweet.Coordinates)))
        |> via' graph
        |> sink writeSink
