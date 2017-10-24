open System
open FsXaml
open ViewModels
open FSharp.Control
open FSharp.Control.Reactive
open PerfUtil
open FSharp.Charting

type App = XAML<"App.xaml">

[<STAThread>]
[<EntryPoint>]
let main argv =

//    let random = new Random(int DateTime.Now.Ticks)
//    let ``500 cities`` = Array.init 500 (fun _ -> random.Next(1001) |> float, random.Next(1001) |> float)
//
//    let tsp = TravelingSalesmanProblem.TravelingSalesmanProblem(100, 0.3, ``500 cities``)
//    let sw = System.Diagnostics.Stopwatch.StartNew()
//    tsp.Execute(30000)
//    |> AsyncSeq.tryLast
//    |> Async.RunSynchronously
//    |> ignore
//    printfn "Time : %d ms to execute %d iterations with %d neurons and %d cities" sw.ElapsedMilliseconds iterations neurons cities.Length
//    Console.WriteLine("End")
//    Console.ReadLine() |> ignore
//    0

    Wpf.installSynchronizationContext ()
    Views.MainWindow()
    |> App().Run
