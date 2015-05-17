module ImagePipeline.Main

open System
open System.Collections.Concurrent
open System.Collections.Generic
open System.Diagnostics.CodeAnalysis
open System.Drawing
open System.IO
open System.Linq
open System.Threading
open System.Threading.Tasks

open ImagePipeline.Image
open ImagePipeline.Extensions
open ImagePipeline.PipelineAgent


type ImageMode =
    | Sequential = 0
    | Pipelined = 1
    | LoadBalanced = 2
    | MessagePassing = 3

let QueueBoundedCapacity = 4
let LoadBalancingDegreeOfConcurrency = 2
let MaxNumberOfImages = 500

[<SuppressMessage("Microsoft.Design", "CA1031:DoNotCatchGeneralExceptionTypes")>]
let imagePipelineMainLoop displayFn token errorFn =
    try
        let sourceDir =  @"C:\Demo\ConcurrecnyFSharp\DemoConcurrency\DemoConcurrency\Data\Images"
        // Ensure that frames are presented in sequence before invoking the user-provided display function.
        let imagesSoFar = ref 0
        let safeDisplayFn (info:ImageInfo) =
            if info.SequenceNumber <> !imagesSoFar then
                failwithf "Images processed out of order. Saw %d, expected %d" info.SequenceNumber (!imagesSoFar)
            displayFn info
            incr imagesSoFar

        // Create a cancellation handle for inter-task signaling of exceptions. This cancellation
        // handle is also triggered by the incoming token that indicates user-requested
        // cancellation.
        use cts = CancellationTokenSource.CreateLinkedTokenSource([| token |]) 
        let fileNames = Utilities.GetImageFilenames sourceDir MaxNumberOfImages        
        runMessagePassing fileNames sourceDir QueueBoundedCapacity safeDisplayFn cts        
    with 
    | :? AggregateException as ae when ae.InnerExceptions.Count = 1 ->
        errorFn (ae.InnerExceptions.[0])
    | :? OperationCanceledException as e -> reraise()
    | e -> errorFn e
