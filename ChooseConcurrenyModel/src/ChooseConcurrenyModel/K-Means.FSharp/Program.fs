open System
open System.Windows.Forms
open FSharp.Charting
open PerfUtil
open KMeans
open KMeans.Data


[<EntryPoint>]
[<STAThread>]
let main argv =

    let M = 7 // number of experiments / random initial centroids
    let initialCentroidsSet =
        [ for i in [1..M] do
            yield data |> getRandomCentroids 11 ]

    let methods =
        ["F# Seq",          (FSharpSeq.kmeans)
         "F# PSeq",         (FSharpPSeq.kmeans)
       //  "F# Streams",      (FSharpStreams.kmeans)
       //  "F# ParStreams",   (FSharpParStreams.kmeans)
         "C# LINQ",         (fun data _ initialCentroids ->
                                KMeans.CSharp.KMeansSimple(data).Run(initialCentroids))
         "C# PLINQ",        (fun data _ initialCentroids ->
                                KMeans.CSharp.KMeansPLinq(data).Run(initialCentroids))
         "C# PLINQ Partitioner", (fun data _ initialCentroids ->
                                    KMeans.CSharp.KMeansPLinqPartitioner(data).Run(initialCentroids))
       //  "C# Stream",       (fun data _ initialCentroids ->
       //                         KMeans.CSharp.KMeansStreams(data).Run(initialCentroids))
       //  "C# ParStream",    (fun data _ initialCentroids ->
       //                         KMeans.CSharp.KMeansParStreams(data).Run(initialCentroids))
        ]
    let perfResults =
        methods
        |> List.map (fun (name,f) ->
            printfn "---------------------------------"
            printfn "Running '%s' implementation ..." name
            let res = Benchmark.Run (fun () ->
                for initialCentroids in initialCentroidsSet do
                    let centroids = f data dist initialCentroids
                    let error = getError centroids
                    //// Positive difference means that out clustering is better than initial one
                    printfn "%f (>0 is good)" (classesError - error)
                    ()
            )
            printfn "\nPerfResult:%A\n" res
            name, res
        )

    let elapsedTimeData =
        perfResults |> List.map (fun (name,x) -> name, x.Elapsed.TotalMilliseconds/float(M))
    let cpuTimeData =
        perfResults |> List.map (fun (name,x) -> name, x.CpuTime.TotalMilliseconds/float(M))
    let getLabels =
        List.map (snd >> sprintf "%.3f")

    let chart =
        [Chart.Column(elapsedTimeData, Name="Elapsed Time (ms)", Labels=(getLabels elapsedTimeData))
         Chart.Column(cpuTimeData,     Name="CPU Time (ms)",     Labels=(getLabels cpuTimeData))]
        |> Chart.Combine
        |> Chart.WithLegend(Title="Elapsed vs CPU Time (ms)")
    Application.Run(chart.ShowChart())

    let plot (initialCentroids:float[][]) =
        methods
        |> List.mapi (fun i (name,f) ->
            printfn "> Running '%s' implementation ..." name
            f data dist initialCentroids
            |> Visualizer.plotCentorids i
            |> Chart.WithLegend(Title=name)
        )
        |> List.chunkBySize 3
        |> List.map (Chart.Columns)
        |> Chart.Rows
    let chart2 = plot (initialCentroidsSet.[0])
    Application.Run(chart2.ShowChart())

    0