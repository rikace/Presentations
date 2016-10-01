module KMeans.FSharpParStreams

open Nessos.Streams
let MaxIterations =1000

let kmeans (data:float[][]) dist initialCentroids =
    let N = data.[0].Length
    let data = ParStream.ofArray data
    let nearestCentroid centroids u =
        Array.minBy (dist u) centroids
    let updateCentroids centroids =
        data
        |> ParStream.groupBy (nearestCentroid centroids)
        |> ParStream.map (fun (_,points) ->
            Array.init N (fun i ->
                points |> Seq.averageBy (fun x-> x.[i]))
        )
        |> ParStream.toArray
        |> Array.sort
    let rec update n centoids =
        let newCentroids = updateCentroids centoids
        let error =
            if centoids.Length <> newCentroids.Length
            then System.Double.MaxValue
            else Array.fold2 (fun x u v -> x + dist u v) 0.0
                        centoids newCentroids
        if n=0 || (error < 1e-9)
        then printfn "Iterations %d" (MaxIterations-n)
             newCentroids
        else update (n-1) newCentroids
    update MaxIterations initialCentroids
