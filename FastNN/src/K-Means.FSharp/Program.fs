open System
open System.Windows.Forms
open FSharp.Charting
open PerfUtil
open KMeans
open KMeans.Data

module Visualizer =

    open KMeans.Data
    open FSharp.Charting
    open Accord.Statistics.Analysis

    let toPairs (data:float[][]) =
        data |> Array.map (fun x->x.[0], x.[1])

    let pca, points =
        let model =
            PrincipalComponentAnalysis(
                Method = PrincipalComponentMethod.Center,
                Whiten = true)
        model.Learn(data) |> ignore
        model.NumberOfOutputs <- 2
        model, model.Transform(data) |> toPairs


    let plotCentorids i (centroids: float[][]) =
        let centroidPoints = pca.Transform(centroids) |> toPairs
        Chart.Combine(
          [Chart.Point(points, MarkerSize=4, Name=sprintf "Data points #%d" i)
           Chart.Point(centroidPoints, MarkerSize=8, Name=sprintf "Centroids #%d" i,
                        Color=System.Drawing.Color.Red)
          ])

[<EntryPoint>]
[<STAThread>]
let main argv =

    let M = 7 // number of experiments / random initial centroids
    let initialCentroidsSet =
        [ for i in [1..M] do
            yield data |> getRandomCentroids 11 ]

    let methods =
        [ //"F# Seq",          (FSharpSeq.kmeans)
         "C# LINQ",         (fun data _ initialCentroids ->
                             KMeans.CSharp.KMeansSimple(data).Run(initialCentroids))]

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