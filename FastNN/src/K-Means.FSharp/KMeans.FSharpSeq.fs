module KMeans.FSharpSeq

// The goal of the dojo will be to
// create a classifier that uses training data
// to recognize hand-written digits, and
// evaluate the quality of our classifier
// by looking at predictions on the validation data.

// This file provides some guidance through the problem:
// each section is numbered, and 
// solves one piece you will need. Sections contain
// general instructions, 
// [ YOUR CODE GOES HERE! ] tags where you should
// make the magic happen, and
// <F# QUICK-STARTER> blocks. These are small
// F# tutorials illustrating aspects of the
// syntax which could come in handy. Run them,
// see what happens, and tweak them to fit your goals!


let MaxIterations =1000

let kmeans (data:float[][]) dist initialCentroids =
    let N = data.[0].Length


    let nearestCentroid centroids u =
        Array.minBy (dist u) centroids

    let updateCentroids (centroids) =
        data
        |> Seq.groupBy (nearestCentroid centroids)
        |> Seq.map (fun (_,points) ->
            Array.init N (fun i ->
                points |> Seq.averageBy (fun x-> x.[i]))
        )
        |> Seq.sort
        |> Seq.toArray

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