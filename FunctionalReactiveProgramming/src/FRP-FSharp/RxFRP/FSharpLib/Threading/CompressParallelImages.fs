namespace Easj360FSharp 

open System.Drawing
open System.IO
open System.Drawing.Imaging
open System.Linq
open Microsoft.FSharp.Control

module CompressParallelImages =

    type ParallelImageResizer(destination:string, quality:System.Int64) =
        let _eventCompleted = new Event<System.ComponentModel.AsyncCompletedEventArgs>()
        let _eventProgres = new Event<System.ComponentModel.ProgressChangedEventArgs>()
                        
        member x.ResizeImages(files:System.IO.FileInfo[]) =
                    let epQuality = new EncoderParameter(System.Drawing.Imaging.Encoder.Quality, quality)
                    let iciCodes = ImageCodecInfo.GetImageEncoders()
                    let iciJpegCodec = iciCodes.First(fun (i:ImageCodecInfo) -> i.MimeType = "image/jpeg")
                    let numeImages = ref 0
                    let epParameters = new EncoderParameters(1);
                    epParameters.Param.[0] <- epQuality                                        
                    files
                    |> Seq.map( fun f -> async { 
                                                let dest = Path.Combine(destination, f.Name)
                                                use newImage = Image.FromFile(f.FullName)                                                 
                                                newImage.Save(dest, iciJpegCodec, epParameters) 
                                                System.Threading.Interlocked.Increment numeImages |> ignore
                                                _eventProgres.Trigger(System.ComponentModel.ProgressChangedEventArgs((System.Threading.Thread.VolatileRead numeImages), dest))
                                                })
                    |> Async.Parallel 
                    |> Async.RunSynchronously
                    |> ignore
                    _eventCompleted.Trigger(System.ComponentModel.AsyncCompletedEventArgs(null,false, "Completed"))    
        
        [<CLIEvent>]
        member x.EventCompleted = _eventCompleted.Publish
        
        [<CLIEvent>]
        member x.EventProgress = _eventProgres.Publish
        

    type AsyncImageResizer(worker:System.Int32, destination:string, quality:System.Int64) =
        let _eventCompleted = new Event<System.ComponentModel.AsyncCompletedEventArgs>()
        let _eventProgres = new Event<System.ComponentModel.ProgressChangedEventArgs>()
        let epQuality = new EncoderParameter(System.Drawing.Imaging.Encoder.Quality, quality)
        let iciCodes = ImageCodecInfo.GetImageEncoders()
        let iciJpegCodec = iciCodes.First(fun (i:ImageCodecInfo) -> i.MimeType = "image/jpeg")
        let epParameters = new EncoderParameters(1);
        let enc = epParameters.Param.[0] <- epQuality         
        let numeImages = ref 0
        let mutable wait = false
        let processImage (f:System.IO.FileInfo) = 
                    async { 
                                let dest = Path.Combine(destination, f.Name)
                                use newImage = Image.FromFile(f.FullName)                                                 
                                newImage.Save(dest, iciJpegCodec, epParameters) 
                                System.Threading.Interlocked.Increment numeImages |> ignore
                                _eventProgres.Trigger(System.ComponentModel.ProgressChangedEventArgs((System.Threading.Thread.VolatileRead numeImages), dest))
                          }
                                  
        let parallelWorker n f = 
            MailboxProcessor.Start(fun inbox ->
                let workers = 
                    Array.init n (fun i -> MailboxProcessor.Start(f))
                let rec loop i = async {                    
                    let! msg = inbox.Receive()
                    workers.[i].Post(msg)
                    return! loop ((i+1) % n)
                }
                loop 0
            )
                     
        let agent = 
            parallelWorker worker (fun inbox ->
            let rec loop() = async {                
                let! msg = inbox.Receive()
                if(wait = true) 
                    then
                        Async.Sleep(75) |> ignore
                        wait <- false
                do! processImage msg
                return! loop()
            }
            loop()
        )
        
        member x.Wait(value:bool) =
                wait <- value
                
        member x.ResizeImages(files:System.IO.FileInfo[]) =
                for f in files do
                    agent.Post(f)
                _eventCompleted.Trigger(System.ComponentModel.AsyncCompletedEventArgs(null,false, "Completed"))    
        
        [<CLIEvent>]
        member x.EventCompleted = _eventCompleted.Publish
        
        [<CLIEvent>]
        member x.EventProgress = _eventProgres.Publish
        
        
        (*
        open System.Drawing;
open System.IO;
open System.Drawing.Imaging;
open System.Linq
open Microsoft.FSharp.Control

module CompressImages =

    type AsyncImageResizer() =
        let _eventCompleted = new Event<System.ComponentModel.AsyncCompletedEventArgs>()
        let _eventProgres = new Event<System.ComponentModel.ProgressChangedEventArgs>()
                
        let agent = 
            MailboxProcessor.Start(fun inbox ->
            let rec loop() = async {                
                let! msg = inbox.Receive()
               
                return! loop()

            }
            loop()
        )
        
        member x.ResizeImages(files:System.IO.FileInfo[], destination:string, quality:System.Int64) =
                    let epQuality = new EncoderParameter(System.Drawing.Imaging.Encoder.Quality, quality)
                    let iciCodes = ImageCodecInfo.GetImageEncoders()
                    let iciJpegCodec = iciCodes.First(fun (i:ImageCodecInfo) -> i.MimeType = "image/jpeg")
                    let numeImages = ref 0
                    let epParameters = new EncoderParameters(1);
                    epParameters.Param.[0] <- epQuality                                        
                    files
                    |> Seq.map( fun f -> async { 
                                                let dest = Path.Combine(destination, f.Name)
                                                use newImage = Image.FromFile(f.FullName)                                                 
                                                newImage.Save(dest, iciJpegCodec, epParameters) 
                                                System.Threading.Interlocked.Increment numeImages |> ignore
                                                _eventProgres.Trigger(System.ComponentModel.ProgressChangedEventArgs((System.Threading.Thread.VolatileRead numeImages), dest))
                                                })
                    |> Async.Parallel 
                    |> Async.
                    |> ignore
                    _eventCompleted.Trigger(System.ComponentModel.AsyncCompletedEventArgs(null,false, "Completed"))    
        
        [<CLIEvent>]
        member x.EventCompleted = _eventCompleted.Publish
        
        [<CLIEvent>]
        member x.EventProgress = _eventProgres.Publish*)