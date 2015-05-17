module ImagePipeline.Image

open System
open System.Diagnostics.CodeAnalysis
open System.Drawing

type ImageInfo(sequenceNumber:int, fileName:string, originalImage:Bitmap, clockOffset:int) =

    let phaseStart = Array.create 4 0
    let phaseEnd = Array.create 4 0

    let mutable queueCount1 = 0
    let mutable queueCount2 = 0
    let mutable queueCount3 = 0
    let mutable imageCount = 0
    let mutable framesPerSecond = 0.0

    let mutable thumbnailImage : Bitmap = null
    let mutable filteredImage : Bitmap = null
    let mutable originalImage = originalImage

    // --------------------------------------------------------------------------
    // Get/set properties that are initialized from the pipeline
    // --------------------------------------------------------------------------
      
    member x.ThumbnailImage 
      with get() = thumbnailImage and set(v) = thumbnailImage <- v
    member x.FilteredImage
      with get() = filteredImage and set(v) = filteredImage <- v
    member x.OriginalImage 
      with get() = originalImage and set(v) = originalImage <- v
    
    // --------------------------------------------------------------------------
    // Properties and state initialized in constructor
    // --------------------------------------------------------------------------

    member x.SequenceNumber = sequenceNumber
    member x.FileName = fileName
    member x.ClockOffset = clockOffset

    [<SuppressMessage("Microsoft.Performance", "CA1819:PropertiesShouldNotReturnArrays")>]
    member x.PhaseStartTick = phaseStart 
    [<SuppressMessage("Microsoft.Performance", "CA1819:PropertiesShouldNotReturnArrays")>]
    member x.PhaseEndTick = phaseEnd 

    // Image pipeline performance data

    member x.QueueCount1 
      with get() = queueCount1 and set(v) = queueCount1 <- v
    member x.QueueCount2 
      with get() = queueCount2 and set(v) = queueCount2 <- v
    member x.QueueCount3 
      with get() = queueCount3 and set(v) = queueCount3 <- v
    member x.ImageCount 
      with get() = imageCount and set(v) = imageCount <- v
    member x.FramesPerSecond 
      with get() = framesPerSecond and set(v) = framesPerSecond <- v

    // --------------------------------------------------------------------------
    // Implementation of the disposable pattern
    // --------------------------------------------------------------------------
    
    member x.Dispose(disposing) =
        if disposing then
            if x.OriginalImage <> null then 
                x.OriginalImage.Dispose()
            if x.ThumbnailImage <> null then 
                x.ThumbnailImage.Dispose()
                x.ThumbnailImage <- null
            if x.FilteredImage <> null then 
                x.FilteredImage.Dispose()
                x.FilteredImage <- null

    interface IDisposable with
        member x.Dispose() =
            x.Dispose(true)
            GC.SuppressFinalize(x)

