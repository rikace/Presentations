#r "bin/Release/FsPickler.dll"
#r "bin/Release/PerfUtil.dll"
#r "bin/Release/PerfUtil.CaseStudy.dll"

#load "../packages/FSharp.Charting.0.90.5/FSharp.Charting.fsx"


open PerfUtil
open PerfUtil.CaseStudy

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
    |> Map.iter (fun _ rs -> plot "milliseconds" (fun r -> r.Elapsed.TotalMilliseconds) rs)

// read performance tests from 'Tests' module and run them
let perfResults =
    PerfTest.OfModuleMarker<Tests.Marker>()
    |> PerfTest.run (fun () -> SerializationPerf.CreateImplementationComparer (warmup = true) :> _)

// plot everything
plotMS perfResults

TestSession.toFile "tests.xml" perfResults
TestSession.ofFile "tests.xml"

// compare performance tests to past versions
let pastPerfResults =
    PerfTest.OfModuleMarker<Tests.Marker> ()
    |> PerfTest.run (fun () -> SerializationPerf.CreatePastVersionComparer (__SOURCE_DIRECTORY__ + "/fspResults.xml") :> _)

// plot everything
plotMS pastPerfResults