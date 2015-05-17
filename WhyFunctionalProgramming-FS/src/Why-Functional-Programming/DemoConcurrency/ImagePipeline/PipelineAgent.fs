module ImagePipeline.PipelineAgent

open System
open System.Threading
open ImagePipeline.Image
open ImagePipeline.BlockingQgent
open System.Collections.Concurrent
open System.Collections.Generic
open ImagePipeline.Extensions
open System.Diagnostics.CodeAnalysis
open System.Drawing
open System.IO
open System.Linq
open System.Threading
open System.Threading.Tasks
open System.Drawing

let GaussianNoiseAmount = 50.0 

let disposeOnException f (obj:#IDisposable) =
    try f obj
    with _ -> 
      obj.Dispose()  
      reraise()

let loadImage fname sourceDir count clockOffset =
    let startTick = Environment.TickCount
    let info = 
        new Bitmap(Path.Combine(sourceDir, fname)) |> disposeOnException (fun bitmap ->
            bitmap.Tag <- fname
            let info = new ImageInfo(count, fname, bitmap, clockOffset)
            info.PhaseStartTick.[0] <- startTick - clockOffset 
            info )
    info.PhaseEndTick.[0] <- Environment.TickCount - clockOffset
    info 


let scaleImage (info:ImageInfo) =
    let startTick = Environment.TickCount
    let orig = info.OriginalImage
    info.OriginalImage <- null
    let scale = 200
    let isLandscape = (orig.Width > orig.Height)
    let newWidth = if isLandscape then scale else scale * orig.Width / orig.Height
    let newHeight = if not isLandscape then scale else scale * orig.Height / orig.Width
    let bitmap = new System.Drawing.Bitmap(orig, newWidth, newHeight)
    try
        bitmap.AddBorder(15) |> disposeOnException (fun bitmap2 ->
            bitmap2.Tag <- orig.Tag
            info.ThumbnailImage <- bitmap2
            info.PhaseStartTick.[1] <- startTick - info.ClockOffset )
    finally
        bitmap.Dispose()
        orig.Dispose()
    info.PhaseEndTick.[1] <- Environment.TickCount - info.ClockOffset


let filterImage (info:ImageInfo) =
    let startTick = Environment.TickCount
    let sc = info.ThumbnailImage
    info.ThumbnailImage <- null
    sc.AddNoise(GaussianNoiseAmount) |> disposeOnException (fun bitmap ->
        bitmap.Tag <- sc.Tag
        info.FilteredImage <- bitmap
        info.PhaseStartTick.[2] <- startTick - info.ClockOffset )
    sc.Dispose()
    info.PhaseEndTick.[2] <- Environment.TickCount - info.ClockOffset


let displayImage (info:ImageInfo) count displayFn duration =
    let startTick = Environment.TickCount
    info.ImageCount <- count
    info.PhaseStartTick.[3] <- startTick - info.ClockOffset
    info.PhaseEndTick.[3] <- 
      if  duration > 0 then startTick - info.ClockOffset + duration
      else Environment.TickCount - info.ClockOffset
    displayFn info
    

let runMessagePassing fileNames sourceDir queueLength displayFn (cts:CancellationTokenSource) = 

  let loadedImages = new BlockingQueueAgent<_>(queueLength)
  let scaledImages = new BlockingQueueAgent<_>(queueLength)    
  let filteredImages = new BlockingQueueAgent<_>(queueLength)    

  // Image pipeline phase 1: Load images from disk and put them a queue.
  let loadImages = async {
    let clockOffset = Environment.TickCount
    let rec numbers n = seq { yield n; yield! numbers (n + 1) }
    for count, img in fileNames |> Seq.zip (numbers 0) do
      let info = loadImage img sourceDir count clockOffset
      do! loadedImages.AsyncAdd(info) }

  // Image pipeline phase 2: Scale to thumbnail size and render picture frame.
  let scalePipelinedImages = async {
    while true do 
      let! info = loadedImages.AsyncGet()
      scaleImage info
      do! scaledImages.AsyncAdd(info) }

  // Image pipeline phase 3: Filter images (give them a 
  // speckled appearance by adding Gaussian noise)
  let filterPipelinedImages = async {
    while true do 
      let! info = scaledImages.AsyncGet()
      filterImage info
      do! filteredImages.AsyncAdd(info) }

  // Image pipeline phase 4: Invoke the user-provided callback 
  // function (for example, to display the result in a UI)
  let displayPipelinedImages = 
    let rec loop count duration = async {
      let! info = filteredImages.AsyncGet()
      let displayStart = Environment.TickCount
      info.QueueCount1 <- loadedImages.Count
      info.QueueCount2 <- scaledImages.Count
      info.QueueCount3 <- filteredImages.Count
      displayImage info (count + 1) displayFn duration
      return! loop (count + 1) (Environment.TickCount - displayStart) }
    loop 0 0

  Async.Start(loadImages, cts.Token)
  Async.Start(scalePipelinedImages, cts.Token)
  Async.Start(filterPipelinedImages, cts.Token)

  try Async.RunSynchronously(displayPipelinedImages, cancellationToken = cts.Token)
  with :? OperationCanceledException -> () 
