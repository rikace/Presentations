module Reactive.StreamTwo

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
open System.Threading.Tasks

type SinkHelper = SinkHelper with
    static member (==>) (_:SinkHelper, a:GraphDsl.ForwardOps<_, NotUsed>) = fun(b:Inlet<string>) -> a.To(b)
    static member (==>) (_:SinkHelper, a:Sink<_,Task>) = fun(b:Source<string, _>) -> b.To(a)
    static member (==>) (_:SinkHelper, a:Source<_,_>) = fun(b:Sink<_,Task>) ->a.To(b)
    static member (==>) (_:SinkHelper, b:Inlet<string>) = fun(a:GraphDsl.ForwardOps<_, NotUsed>) -> a.To(b)
let inline sink x = SinkHelper ==> x

type MatHelpler = MatHelpler with
    //static member (==>) (_:MatHelpler, a:IGraph<FlowShape<_, _>, NotUsed>) = fun (b:GraphDsl.ForwardOps<_, NotUsed>) -> b.Via(a)
    static member (==>) (_:MatHelpler, a:IGraph<FlowShape<_, _>, NotUsed>) = fun (b:Source<ITweet, _>) -> b.Via(a)
let inline mat x = MatHelpler ==> x

type ViaHelper = ViaHelper with
    static member (==>) (_:ViaHelper, a:IGraph<FlowShape<_, _>, NotUsed>) = fun (b:GraphDsl.ForwardOps<_, NotUsed>) -> b.Via(a)
    static member (==>) (b:ViaHelper, a:GraphDsl.ForwardOps<_, NotUsed>) = fun (x:IGraph<FlowShape<_, _>, NotUsed>) -> a.Via(x)
    static member (==>) (b:ViaHelper, a:Source<ITweet, _>) = fun(x:IGraph<FlowShape<_, _>,_>) -> a.Via(x)
let inline via x = ViaHelper ==> x


let inline select map (flow:Flow<_,_, NotUsed>) =
    flow.Select(new Func<_,_>(map))

let inline where (predicate:Predicate<_>) (source:Source<_,_>) = source.Where(predicate)

let inline throttle (elements:int) (per:TimeSpan) (maximumBurst:int) (mode:ThrottleMode) (flow:Flow<_,_,_>) =
    flow.Throttle(elements, per, maximumBurst, mode)

module TweetsWithThrottle =

    let create<'a>(tweetSource:Source<ITweet, 'a>) =
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

        let graph = GraphDsl.Create(fun buildBlock ->
            let broadcast = buildBlock.Add(Broadcast<ITweet>(2))
            let merge = buildBlock.Add(Merge<string>(2))

            buildBlock.From(broadcast.Out(0))
            |> via (flowCreateBy |> throttle 10 (TimeSpan.FromSeconds(1.)) 1 ThrottleMode.Shaping)
            |> via formatUser
            |> sink (merge.In(0))
            |> ignore

            buildBlock.From(broadcast.Out(1))
            |> via (flowCoordinates
                                    //.Buffer(10, OverflowStrategy.DropNew)
                        |> throttle 1 (TimeSpan.FromSeconds(1.)) 10 ThrottleMode.Shaping)
            |> via formatCoordinates
            |> sink (merge.In(1))
            |> ignore

            FlowShape<ITweet, string>(broadcast.In, merge.Out))

        tweetSource
        |> where (Predicate(fun (tweet:ITweet) -> not(isNull tweet.Coordinates)))
        |> mat graph
        |> sink writeSink

module TweetsWeatherWithThrottle =
    open System.Threading.Tasks

    let create<'a>(tweetSource:Source<ITweet, 'a>) : IRunnableGraph<'a> =

        let formatUser =
            Flow.Create<IUser>()
            |> select (Utils.FormatUser)

        let formatCoordinates =
            Flow.Create<ICoordinates>()
            |> select (Utils.FormatCoordinates)

        let formatTemperature =
            Flow.Create<decimal>()
            |> select (Utils.FormatTemperature)

        let createBy = Flow.Create<ITweet>().Select(fun tweet -> tweet.CreatedBy)

        let xn s = XName.Get s
        let getWeatherAsync(coordinates:ICoordinates) =
            async {
                use httpClient = new System.Net.WebClient()
                let requestUrl = sprintf "http://api.met.no/weatherapi/locationforecast/1.9/?lat=%f;lon=%f" coordinates.Latitude coordinates.Latitude
                printfn "%s" requestUrl

                let! result = httpClient.AsyncDownloadString (Uri requestUrl)
                let doc = XDocument.Parse(result)
                let temp = doc.Root.Descendants(xn "temperature").First().Attribute(xn "value").Value
                return Decimal.Parse(temp)
            } |> Async.StartAsTask

        let writeSink =
            Sink.ForEach<string>(fun msg -> Console.WriteLine(msg))

                // 1- Throttle line 72 (Throttle(1))
                // 2- Throttle line 72 (Throttle(10) >> same rate or??
                //    only 1 message because we have 1 stream source & broadcast = 2 channel
                // with 1 request with 10 msg per second and 1 request with 1 msg per second...
                // but we have only 1 stream source, so it cannot send messages to a
                // different rate thus it satisfy the lowest requirement.
        let selectAsync (parallelism:int) (asyncMapper:_ -> Task<_>) (flow:Flow<_, _,NotUsed>) =
            flow.SelectAsync(parallelism, Func<_, Task<_>>(asyncMapper))

        let graph = GraphDsl.Create(fun buildBlock ->
            let broadcast = buildBlock.Add(Broadcast<ITweet>(2))
            let merge = buildBlock.Add(Merge<string>(2))

            buildBlock.From(broadcast.Out(0))
            |> via (Flow.Create<ITweet>()
                    |> select (fun tweet -> tweet.CreatedBy)
                    |> throttle 10 (TimeSpan.FromSeconds(1.)) 1 ThrottleMode.Shaping)
            |> via (formatUser)
            |> sink (merge.In(0))
            |> ignore

            buildBlock.From(broadcast.Out(1))
            |> via (Flow.Create<ITweet>()
                    |> select (fun tweet -> tweet.Coordinates)
                    |> throttle 1 (TimeSpan.FromSeconds(1.)) 1 ThrottleMode.Shaping)
                    |> via (Flow.Create<ICoordinates>()
                             |> selectAsync 5 (fun c -> Utils.GetWeatherMemoizeAsync(c)))// getWeatherAsync(c)))
            |> via (formatTemperature)
            |> sink (merge.In(1))
            |> ignore

            FlowShape<ITweet, string>(broadcast.In, merge.Out))

        tweetSource
        |> where (Predicate(fun (tweet:ITweet) -> not(isNull tweet.Coordinates)))
        |> mat graph
        |> sink writeSink

module TweetsWithWeather =
    open System.Threading.Tasks

    let xn s = XName.Get s
    let getWeatherAsync(coordinates:ICoordinates) =
        async {
            use httpClient = new System.Net.WebClient()
            let requestUrl = sprintf "http://api.met.no/weatherapi/locationforecast/1.9/?lat=%f;lon=%f" coordinates.Latitude coordinates.Latitude
            printfn "%s" requestUrl

            let! result = httpClient.AsyncDownloadString (Uri requestUrl)
            let doc = XDocument.Parse(result)
            let temp = doc.Root.Descendants(xn "temperature").First().Attribute(xn "value").Value
            return Decimal.Parse(temp)
        } |> Async.StartAsTask


    let create<'a>(tweetSource:Source<ITweet, 'a>) =

        let buffer (size:int) (strategy:OverflowStrategy) (flow:Flow<_,_,NotUsed>) =
            flow.Buffer(10, strategy)

        let formatUser =
            Flow.Create<IUser>()
            |> select (Utils.FormatUser)

        let formatCoordinates =
            Flow.Create<ICoordinates>()
            |> select (Utils.FormatCoordinates)

        let formatTemperature =
            Flow.Create<decimal>()
            |> select (Utils.FormatTemperature)

        let createBy = Flow.Create<ITweet>().Select(fun tweet -> tweet.CreatedBy)

        let writeSink =
            Sink.ForEach<string>(fun msg -> Console.WriteLine(msg))

        let selectAsync (parallelism:int) (asyncMapper:_ -> Task<_>) (flow:Flow<_, _,NotUsed>) =
            flow.SelectAsync(parallelism, Func<_, Task<_>>(asyncMapper))


        let graph = GraphDsl.Create(fun buildBlock ->
            let broadcast = buildBlock.Add(Broadcast<ITweet>(2))
            let merge = buildBlock.Add(Merge<string>(2))

            buildBlock.From(broadcast.Out(0))
            |> via (Flow.Create<ITweet>()
                    |> select (fun tweet -> tweet.CreatedBy)
                    |> throttle 10 (TimeSpan.FromSeconds(1.)) 1 ThrottleMode.Shaping)
            |> via (formatUser)
            |> sink (merge.In(0))
            |> ignore

            buildBlock.From(broadcast.Out(1))
            |> via (Flow.Create<ITweet>()
                    |> select (fun tweet -> tweet.Coordinates)
                    |> buffer 10 OverflowStrategy.DropNew
                    |> throttle 1 (TimeSpan.FromSeconds(1.)) 1 ThrottleMode.Shaping)
                    |> via (Flow.Create<ICoordinates>()
                             |> selectAsync 5
                                (fun c -> Utils.Cache(Func<_,_>(Utils.GetWeatherMemoizeAsync)).Invoke(c)))
            |> via (formatTemperature)
            |> sink (merge.In(1))
            |> ignore

            FlowShape<ITweet, string>(broadcast.In, merge.Out))

        tweetSource
        |> where (Predicate(fun (tweet:ITweet) -> not(isNull tweet.Coordinates)))
        |> mat graph
        |> sink writeSink

