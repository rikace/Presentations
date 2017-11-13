open System
open System.Collections.Generic
open System.Configuration
open Akka
open Akka.Actor
open Akka.Streams
open Tweetinvi
open Tweetinvi.Models
open Akka.Streams.Dsl
open Shared.Reactive
open Shared.Reactive.Tweets
open Reactive.StreamOne
open Reactive.StreamTwo
open Reactive.Stream.TweetsToEmotion

type RunnableGraphType =
    | TweetsToConsole
    | TweetsWithBroadcast
    | TweetsWithThrottle
    | TweetsWithWeather
    | TweetsWithWeatherThrottle
    | TweetsToEmotion

module Graph =
    let inline graph<'a>(tweetSource:Source<ITweet, 'a>) grapType =
        match grapType with
        | RunnableGraphType.TweetsToConsole -> TweetsToConsole.create(tweetSource)
        | RunnableGraphType.TweetsWithBroadcast -> TweetsWithBroadcast.create(tweetSource)
        | RunnableGraphType.TweetsWithThrottle -> TweetsWithThrottle.create(tweetSource)
        | RunnableGraphType.TweetsWithWeather -> TweetsWithWeather.create(tweetSource)
        | RunnableGraphType.TweetsWithWeatherThrottle -> TweetsWeatherWithThrottle.create(tweetSource)
        | RunnableGraphType.TweetsToEmotion -> Reactive.Stream.TweetsToEmotion.create(tweetSource)


[<EntryPoint;STAThread>]
let main argv =

    use system = ActorSystem.Create("Reactive-System")
    let consumerKey = ConfigurationManager.AppSettings.["ConsumerKey"]
    let consumerSecret = ConfigurationManager.AppSettings.["ConsumerSecret"]
    let accessToken = ConfigurationManager.AppSettings.["AccessToken"]
    let accessTokenSecret = ConfigurationManager.AppSettings.["AccessTokenSecret"]

    Console.OutputEncoding <- System.Text.Encoding.UTF8
    Console.ForegroundColor <- ConsoleColor.Cyan

    Console.WriteLine("<< Press Enter to Start >>")
    Console.ReadLine() |> ignore

    let useCachedTweets = true
    let grapType = RunnableGraphType.TweetsWithWeather //.TweetsWithWeatherThrottle

    use materialize = system.Materializer()


    if useCachedTweets then
        let tweetSource = Source.FromEnumerator(fun () -> (new TweetEnumerator(true)) :> IEnumerator<ITweet>)
        let graph = Graph.graph<NotUsed>(tweetSource) grapType
        graph.Run(materialize) |> ignore

    else
        Auth.SetCredentials(new TwitterCredentials(consumerKey, consumerSecret, accessToken, accessTokenSecret))

        let tweetSource = Source.ActorRef<ITweet>(100, OverflowStrategy.DropBuffer)
        let graph = Graph.graph<IActorRef>(tweetSource) grapType
        let actor = graph.Run(materialize)

        Utils.StartSampleTweetStream(actor)


    Console.WriteLine("<< Press Enter to Exit >>")

    Reactive.Stream.TweetsToEmotion.showForm()

    Console.ReadLine() |> ignore
    0
