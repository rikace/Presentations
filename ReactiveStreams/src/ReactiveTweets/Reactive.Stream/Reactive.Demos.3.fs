module Reactive.Stream

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

let inline via< ^a, ^b, ^c when ^a : (member Via: ^b -> ^c )> (b: ^b) a =
      (^a : (member Via: ^b -> ^c) (a, b))


//let inline from< ^a, ^b, ^c when ^a : (member From: ^b -> ^c )> (b: ^b) a =
//      (^a : (member From: ^b -> ^c) (a, b))


module TweetsToEmotion =
    open System.Threading.Tasks
    open Reactive.Emotion
    open Tweetinvi.Models
    open LiveCharts
    open LiveCharts.WinForms
    open System.Windows.Forms
    open System.Windows.Threading

    let inline sink b (a: ^a) = (^a : (member To: ^b -> ^c) (a, b))
    //let inline sink b = (^a : (static member To: ^b -> ^c) b)

    let createPieSeries(title:string) : LiveCharts.Wpf.PieSeries =
        let series = new LiveCharts.Wpf.PieSeries()
        series.Title <- title
        series.Values <- new ChartValues<int>(Seq.singleton 0)
        series.DataLabels <- true
        series

    let doughnutChart() : PieChart =
        let pieChart = new PieChart()
        pieChart.InnerRadius <-100.
        pieChart.LegendLocation <- LegendLocation.Right
        pieChart.Series <- new SeriesCollection()
        pieChart.Series.Add(createPieSeries("Unhappy"))
        pieChart.Series.Add(createPieSeries("Indifferent"))
        pieChart.Series.Add(createPieSeries("Happy"))
        pieChart

    let mutable form = Unchecked.defaultof<Form>

    let showForm() =
        if isNull form then ()
        else
            form.ShowDialog() |> ignore

    let initForm(control:Control) =
        form <- new Form ( Text = "Emotions", Width = 400, Height = 300 )
        control.Dock <- DockStyle.Fill
        form.Controls.Add(control)
        let dispatcher = Dispatcher.CurrentDispatcher
        (form, dispatcher)

    let updateChart (chart:PieChart) (emotion:EmotionType) (dispatcher:Dispatcher) =
        dispatcher.Invoke(fun () ->
            let series =
                let series = chart.Series |> Seq.find(fun x -> x.Title = (string emotion))
                series.Values
            series.[0] <- 1 + (series.[0] |> unbox))


    let create<'a>(tweetSource:Source<ITweet, 'a>) =

        let formatFlow =
            Flow.Create<ITweet>()
            |> select Analysis.addEmotion


        let chart = doughnutChart()
        let (form, dispatcher) = initForm(chart)

        let writeSink = Sink.ForEach<ITweet * EmotionType>(fun (tweet, emotion) ->
                updateChart (chart) emotion dispatcher
                printfn "[ %s ] - Tweet [ %s ]" (string emotion) tweet.Text )

        tweetSource
        |> via formatFlow
        |> sink writeSink




