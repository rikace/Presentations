#I "bin"
#r "Streams.Core.dll"
#r "System.Reactive.Core.dll"                  
#r "System.Reactive.Linq.dll"                  
#r "System.Reactive.Interfaces.dll"                  
#load "SimpleStreams.fs"
#load "StreamsEx.fs"
#load "ObservableEx.fs"


open System
open System.Reactive
open System.Reactive.Linq
open SimpleStreams
open ObservableEx

let data = [| 1..10000000 |] |> Array.map (fun x -> x % 1000) |> Array.map int64

let log s f = 
    GC.Collect()
    GC.WaitForPendingFinalizers()
    GC.Collect()
        
    let sw = System.Diagnostics.Stopwatch.StartNew()
    let res = f()
    printfn "%s - Result %A - Completed in %s ms" s res (sw.ElapsedMilliseconds.ToString())

// #time;;
let rxValue() =
   data
      .ToObservable()
      .Where(fun x -> x%2L = 0L)
      .Select(fun x -> x * x)
      .Sum()      
      .ToEnumerable()
      |> Seq.head

let streamValue() =
   data
   |> Stream.ofArray
   |> Stream.filter (fun x -> x%2L = 0L)
   |> Stream.map (fun x -> x * x)
   |> Stream.sum
   
let obsValue() =
   data
   |> ObservableEx.ofSeq
   |> Observable.filter (fun x -> x%2L = 0L)
   |> Observable.map (fun x -> x * x)
   |> ObservableEx.sum
   |> ObservableEx.first


log "RX" rxValue
log "Streams" streamValue
log "obsValue" obsValue