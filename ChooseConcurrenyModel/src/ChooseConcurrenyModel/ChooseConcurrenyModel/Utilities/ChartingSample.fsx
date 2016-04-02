module _2

open System.Threading

#load "FSharp.Charting.fsx"
open FSharp.Charting
open System

let showChart (chart:ChartTypes.GenericChart) = chart.ShowChart()

// Drawing graph of a 'square' function 
let chart = Chart.Line [ for x in 1.0 .. 100.0 -> (x, x ** 2.0) ] |> showChart

Chart.Line [ for x in 1.0 .. 100.0 -> x * x * sin x ] |> showChart


let rnd = new Random()
let rand() = rnd.NextDouble()
let futureDate numDays = DateTime.Today.AddDays(float numDays)

let expectedIncome = [ for x in 1 .. 100 -> (futureDate x, 1000.0 + rand() * 100.0 * exp (float x / 40.0) ) ]
let expectedExpenses = [ for x in 1 .. 100 -> (futureDate x, rand() * 500.0 * sin (float x / 50.0) ) ]
let computedProfit = (expectedIncome, expectedExpenses) ||> List.map2 (fun (d1,i) (d2,e) -> (d1, i - e))

Chart.Line(expectedIncome,Name="Income")
Chart.Line(expectedExpenses,Name="Expenses")
Chart.Line(computedProfit,Name="Profit")

Chart.Combine(
   [ Chart.Line(expectedIncome,Name="Income")
     Chart.Line(expectedExpenses,Name="Expenses") 
     Chart.Line(computedProfit,Name="Profit") ]).ShowChart()module ChartingSample

