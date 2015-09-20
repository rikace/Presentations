#I "bin"
#r "Streams.Core.dll"
#r "FSharp.Collections.ParallelSeq.dll"
#r "System.Reactive.Core.dll"
#r "System.Reactive.Linq.dll"
#r "PerfUtil.dll"

open Nessos.Streams
open System

#time

let data = [| 1..10000000 |] |> Array.map int64

Stream.ofArray data
|> Stream.filter (fun v -> v % 2L = 0L)
|> Stream.map (fun v -> v + 1L)
|> Stream.sum

// Switching to the full Streams library, there’s support for parallel streams via the ParStream module:
ParStream.ofArray data
|> ParStream.filter (fun v -> v % 2L = 0L)
|> ParStream.map (fun v -> v + 1L)
|> ParStream.sum

open FSharp.Collections.ParallelSeq

PSeq.ofArray data
|> PSeq.filter (fun v -> v % 2L = 0L)
|> PSeq.map (fun v -> v + 1L)
|> PSeq.sum

data
|> Seq.filter (fun v -> v % 2L = 0L)
|> Seq.map (fun v -> v + 1L)
|> Seq.sum

let seqValue = 
   data
   |> Seq.filter (fun x -> x%2L = 0L)
   |> Seq.map (fun x -> x * x)
   |> Seq.sum

let streamValue =
   data
   |> Stream.ofArray
   |> Stream.filter (fun x -> x%2L = 0L)
   |> Stream.map (fun x -> x * x)
   |> Stream.sum

// For operations over arrays, the F# Array module would be more appropriate choice and is slightly faster
let arrayValue =
   data
   |> Array.filter (fun x -> x%2L = 0L)
   |> Array.map (fun x -> x * x)
   |> Array.sum

// F# Interactive running in 64-bit mode Streams take back the advantage
// Looks like the 64-bit JIT is doing some black magic there.

let streamValue =
   data
   |> Stream.ofArray
   |> Stream.filter (fun x -> x%2L = 0L)
   |> Stream.map (fun x -> x * x)
   |> Stream.sum



#load "ObservableEx.fs"
#I "bin"
#r "Streams.Core.dll"
#r "System.Reactive.Core.dll"                  
#r "System.Reactive.Linq.dll"                  
#r "System.Reactive.Interfaces.dll"

open System.Reactive.Linq
open SimpleStreams
open ObservableEx

// Streams library and Rx have different goals (data processing vs event processing)
// but I thought it would be interesting to compare them all the same
let rxValue =
   data
      .ToObservable()
      .Where(fun x -> x%2L = 0L)
      .Select(fun x -> x * x)
      .Sum()      
      .ToEnumerable()
      |> Seq.head

let streamValue' =
   data
   |> Stream.ofArray
   |> Stream.filter (fun x -> x%2L = 0L)
   |> Stream.map (fun x -> x * x)
   |> Stream.sum

let obsValue =
   data
   |> ObservableEx.ofSeq
   |> Observable.filter (fun x -> x%2L = 0L)
   |> Observable.map (fun x -> x * x)
   |> ObservableEx.sum
   |> ObservableEx.first



open PerfUtil
open System.Threading
let result = Benchmark.Run(fun () -> Thread.Sleep 10)