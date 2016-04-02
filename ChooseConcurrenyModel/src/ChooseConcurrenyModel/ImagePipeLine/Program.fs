open Microsoft.FSharp.Control    
open ImagePipeline
open System
open System.IO
open System.Drawing
open System.Threading
open AsyncBoundedQueue
open CommonHelpers
open Common
open ImageProcessing

[<EntryPoint>]
let main argv = 

    let cts = new System.Threading.CancellationTokenSource()

    let pipeLineImagesTest = PipeLineImages(4, cts)

    let processImagesPipeline() =
        Async.Start(pipeLineImagesTest.Step1_loadImages, cts.Token) // STEP 1
        Async.Start(pipeLineImagesTest.Step2_scalePipelinedImages, cts.Token) // STEP 2
        Async.Start(pipeLineImagesTest.Step3_filterPipelinedImages, cts.Token) // STEP 3
        Async.Start(pipeLineImagesTest.Step4_displayPipelinedImages, cts.Token) // STEP 4

    StartPhotoViewer.start() |> ignore

    Console.WriteLine("<< press ENTER to start >>")
    Console.ReadLine() |> ignore

    processImagesPipeline() // Pipeline-Agent

    Console.WriteLine("<< press ENTER to cancel >>")
    Console.ReadLine() |> ignore

    cts.Cancel()

    printfn "%A" argv
    0 // return an integer exit code
