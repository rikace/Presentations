module SimpleChart

open System
open FSharp.Charting
open System.Reactive.Linq


let list = [1..10]
let squereList = list |> List.map (fun x -> x * x)
Chart.Combine(
   [ Chart.Line(list,Name="list")
     Chart.Line(squereList,Name="squereList") ])
|> Chart.WithLegend true
|> Chart.Show

let obs1 = Observable
            .Interval(TimeSpan.FromMilliseconds(10.0))
            .Select(fun x -> let rad = (float x)*Math.PI/180.0
                             cos rad, sin rad)
            .Take(360)

let obs2 = Observable
            .Interval(TimeSpan.FromMilliseconds(10.0))
            .Select(fun x -> let rad = (float x)*Math.PI/180.0
                             cos (rad*5.0), sin (rad*7.0))
            .Take(360)

let obsm = obs1 |> Observable.merge obs2 

Chart.Combine(
    [
        LiveChart.LineIncremental obsm
    ])
|> Chart.WithLegend true
|> Chart.Show