#I "bin"
#r "Streams.Core.dll"
#r "FSharp.Collections.ParallelSeq.dll"

open Nessos.Streams
open FSharp.Collections.ParallelSeq

let data = [| 1..10000000 |] |> Array.map (fun x -> x % 1000) |> Array.map int64
let dataHigh = [| 1..1000000 |] |> Array.map int64
let dataLow = [| 1..10 |] |> Array.map int64

let sumSeq() = Seq.sum data
let sumArray () = Array.sum data
let sumStreams () = Stream.ofArray data |> Stream.sum
//let sumLinqOpt () = Query.ofSeq data |> Query.sum |> Query.compile

let sumSqSeq () = data |> Seq.map (fun x -> x * x) |> Seq.sum
let sumSqArray () = data |> Array.map (fun x -> x * x) |> Array.sum
let sumSqStreams () = Stream.ofArray data |> Stream.map (fun x -> x * x) |> Stream.sum
//let sumSqLinqOpt = v |> Query.ofSeq |> Query.map(fun x -> x * x) |> Query.sum |> Query.compile

let sumSqEvenSeq () = data |> Seq.filter (fun x -> x % 2L = 0L) |> Seq.map (fun x -> x * x) |> Seq.sum
let sumSqEvenArray() = data |> Array.filter (fun x -> x % 2L = 0L) |> Array.map (fun x -> x * x) |> Array.sum
let sumSqEvenStreams() = Stream.ofArray data |> Stream.filter (fun x -> x % 2L = 0L) |> Stream.map (fun x -> x * x) |> Stream.sum
//let sumSqEvenLinqOpt = v |> Query.ofSeq |> Query.filter (fun x -> x % 2L = 0L) |> Query.map(fun x -> x * x) |> Query.sum |> Query.compile

let cartSeq () = dataHigh |> Seq.collect (fun x -> Seq.map (fun y -> x * y) dataLow) |> Seq.sum
let cartArray () = dataHigh |> Array.collect (fun x -> Array.map (fun y -> x * y) dataLow) |> Array.sum
let cartStreams () = Stream.ofArray dataHigh |> Stream.collect (fun x -> Stream.ofArray dataLow |> Stream.map (fun y -> x * y)) |> Stream.sum
//let cartLinqOpt = vHi |> Query.ofSeq |> Query.collect (fun x -> Seq.map (fun y -> x * y) vLow) |> Query.sum |> Query.compile

//let parallelSumLinqOpt= v |> PQuery.ofSeq |> PQuery.sum |> PQuery.compile

let parallelSumSqSeq() = data |> PSeq.map (fun x -> x * x) |> PSeq.sum
let parallelSumSqStreams () = data |> ParStream.ofArray |> ParStream.map (fun x -> x * x) |> ParStream.sum

let parallelSumSqEvenSeq () = data |> PSeq.filter (fun x -> x % 2L = 0L) |> PSeq.map (fun x -> x * x) |> PSeq.sum
let parallelSumSqEvenStreams () = ParStream.ofArray data |> ParStream.filter (fun x -> x % 2L = 0L) |> ParStream.map (fun x -> x * x) |> ParStream.sum

let parallelCartSeq () = dataHigh |> PSeq.collect (fun x -> Seq.map (fun y -> x * y) dataLow) |> PSeq.sum
let parallelCartStreams () = ParStream.ofArray dataHigh |> ParStream.collect (fun x -> Stream.ofArray dataLow |> Stream.map (fun y -> x * y)) |> ParStream.sum


#r @"PerfUtil.dll"

open PerfUtil

[<AbstractClass>]
type PerfTests(name) =
    
    abstract Sum: unit -> unit
    abstract SumSq: unit -> unit
    abstract SumSqEven: unit -> unit
    abstract Cart: unit -> unit
    abstract ParallelSumSq: unit -> unit
    abstract ParallelSumSqEven: unit -> unit
    abstract ParallelCart: unit -> unit

    interface ITestable with
        override __.Name = name
        override __.Init() = ()
        override __.Fini() = ()


type SeqTests() =
    inherit PerfTests("Seq")

    override __.Sum() = sumSeq() |> ignore
    override __.SumSq() = sumSqSeq() |> ignore
    override __.SumSqEven() = sumSqEvenSeq() |> ignore
    override __.Cart() = cartSeq() |> ignore
    override __.ParallelSumSq() = parallelSumSqSeq() |> ignore
    override __.ParallelSumSqEven() = parallelSumSqEvenSeq() |> ignore
    override __.ParallelCart() = parallelCartSeq() |> ignore

let seqTests = new SeqTests() :> PerfTests

type ArrayTests() =
    inherit PerfTests("Array")

    override __.Sum() = sumArray() |> ignore
    override __.SumSq() = sumSqArray() |> ignore
    override __.SumSqEven() = sumSqEvenArray() |> ignore
    override __.Cart() = cartArray() |> ignore
    override __.ParallelSumSq() = ()
    override __.ParallelSumSqEven() = ()
    override __.ParallelCart() = ()

let arrayTests = new ArrayTests() :> PerfTests

type StreamsTests() =
    inherit PerfTests("Streams")

    override __.Sum() = sumStreams() |> ignore
    override __.SumSq() = sumSqStreams() |> ignore
    override __.SumSqEven() = sumSqEvenStreams() |> ignore
    override __.Cart() = cartStreams() |> ignore
    override __.ParallelSumSq() = parallelSumSqStreams() |> ignore
    override __.ParallelSumSqEven() = parallelSumSqEvenStreams() |> ignore
    override __.ParallelCart() = parallelCartStreams() |> ignore

let streamsTests = new StreamsTests() :> PerfTests

type Tests =
    [<PerfTest(10)>]
    static member Sum (tests: PerfTests) = tests.Sum()

    [<PerfTest(10)>]
    static member ``Sum of Squares`` (tests: PerfTests) = tests.SumSq()

    [<PerfTest(10)>]
    static member ``Sum of Squares Even`` (tests: PerfTests) = tests.SumSqEven()

    [<PerfTest(10)>]
    static member ``Cartesian Product`` (tests: PerfTests) = tests.Cart()

    [<PerfTest(10)>]
    static member ``Parallel Sum of Squares`` (tests: PerfTests) = tests.ParallelSumSq()

    [<PerfTest(10)>]
    static member ``Parallel Sum of Squares Even`` (tests: PerfTests) = tests.ParallelSumSqEven()

    [<PerfTest(10)>]
    static member ``Parallel Cartesian Product`` (tests: PerfTests) = tests.ParallelCart()

let tests = PerfTest<PerfTests>.OfType<Tests>()

let createStreamsImplemenationCompararer() =
    new ImplementationComparer<_>(streamsTests, [seqTests; arrayTests]) :> PerformanceTester<_>

tests |> PerfTest.run (fun () -> createStreamsImplemenationCompararer())

#load "..\..\StreamsCps\packages\FSharp.Charting.0.90.12\FSharp.Charting.fsx"

open FSharp.Charting

// simple plot function
let plot yaxis (metric : PerfResult -> float) (results : PerfResult list) =
    let values = results |> List.choose (fun r -> if r.HasFailed then None else Some (r.SessionId, metric r))
    let name = results |> List.tryPick (fun r -> Some r.TestId)
    let ch = Chart.Bar(values, ?Name = name, ?Title = name, YTitle = yaxis)
    ch.ShowChart()

// plot milliseconds
let plotMS (results : TestSession list) = 
    results 
    |> TestSession.groupByTest
    |> Map.iter (fun _ rs -> plot "milliseconds" (fun r -> r.Elapsed.TotalMilliseconds) rs |> ignore)

// read performance tests from 'Tests' type and run them
let perfResults =
    PerfTest.OfType<Tests>()
    |> PerfTest.run (fun () -> createStreamsImplemenationCompararer())

// plot everything
plotMS perfResults

TestSession.toFile "tests.xml" perfResults
TestSession.ofFile "tests.xml"


