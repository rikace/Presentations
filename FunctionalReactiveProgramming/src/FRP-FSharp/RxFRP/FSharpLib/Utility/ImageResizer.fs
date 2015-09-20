namespace Easj360FSharp

open System.Drawing
open System.IO
open System.Drawing.Imaging
open System.Linq
open System.Threading

module ModuleImage =

type MMMM =
    | SomeThing of string

open System
open System.IO;
open System.Drawing
open System.Drawing.Drawing2D
open System.Drawing.Imaging
//#r "System.Drawing"

let resizeImage (size:Size) (format:ImageFormat) (source:string) (destination:string) = async {
    let image = Image.FromFile(source)
    let sourceWidth = image.Width
    let sourceHeight = image.Height
    let nPercentW = float(size.Width) / float(sourceWidth) 
    let nPercentH = float(size.Height) / float(sourceHeight)
    let nPercent = if nPercentH < nPercentW then nPercentH
                   else nPercentW
    let destWidth = sourceWidth * int(nPercent)
    let destHeight = sourceHeight * int(nPercent)
    use bitmap = new Bitmap(destWidth, destWidth)
    use g = Graphics.FromImage(bitmap :> Image)
    g.InterpolationMode <- InterpolationMode.HighQualityBicubic
    g.DrawImage(image, 0, 0, destWidth, destHeight)
    bitmap.Save(destination, format) }
let isImage imageFile =
    match Path.GetExtension(imageFile) with
    | ".jpg"
    | ".bmp"
    | ".gif"
    | ".png"
    | ".tiff"
    | ".jpeg" -> true
    | _ -> false
let getImages sourceFolder destinationFolder=
    let images = Directory.EnumerateFiles(sourceFolder)
    let resizeImageCurry = resizeImage (new Size(48,29)) ImageFormat.Jpeg     
    images
    |> Seq.filter isImage
    |> Seq.map (fun file -> resizeImageCurry file (Path.Combine(destinationFolder, Path.GetFileNameWithoutExtension(file))))
    |> Async.Parallel
    |> Async.Ignore
    |> Async.RunSynchronously
 

//    type Resizer() = 
//        val source: string
//        val destination: string
//        val mutable pattern: string
//        val private completed : Event<string>
//        val private syncCtx : SynchronizationContext
//        val private asyncGate : AsyncGate.RequestGate
//
//        new(source, destination) =
//            {source = source; destination = destination; pattern = System.String.Empty; 
//             completed = new Event<string>(); syncCtx = SynchronizationContext.Current;
//             asyncGate = new  AsyncGate.RequestGate(System.Environment.ProcessorCount)}
//
//        member x.Patterns
//            with get() = x.pattern
//            and set(value) = x.pattern <- value
//
//        [<CLIEvent>]
//        member x.Completed = x.completed.Publish
//
//        member private x.RaiseEventOnGuiThread (event:Event<_>) args =
//            try
//                match x.syncCtx with 
//                | null -> x.completed.Trigger(args)
//                | s -> s.Post(SendOrPostCallback(fun _ -> event.Trigger args),state=null)
//            with
//            |_ -> ()
//
//
//        member private x.ParallelWorker n f = 
//            MailboxProcessor.Start(fun inbox ->
//                let workers = 
//                    Array.init n (fun i -> MailboxProcessor.Start(f))
//                let rec loop i = async {                
//                    let! msg = inbox.Receive()
//                    workers.[i].Post(msg)
//                    return! loop ((i+1) % n)
//                }
//                loop 0
//            )
//
//        member x.Agent = 
//                x.ParallelWorker System.Environment.ProcessorCount (fun inbox ->
//                    let epQuality = new EncoderParameter(System.Drawing.Imaging.Encoder.Quality, 25L)
//                    let epCompression = new EncoderParameter(System.Drawing.Imaging.Encoder.Compression, 25L)
//                    let iciCodes = ImageCodecInfo.GetImageEncoders()
//                    let iciJpegCodec = iciCodes.First(fun (i:ImageCodecInfo) -> i.MimeType = "image/jpeg")
//                    let epParameters = new EncoderParameters(1);
//                    epParameters.Param.[0] <- epQuality
//                    let checkDir (dir:System.IO.DirectoryInfo) = 
//                        if not dir.Exists then
//                            dir.Create()                    
//                    let checkFile (f:System.IO.FileInfo) =  
//                        let destinationDir = f.Directory
//                        let destinationFilePath = System.IO.DirectoryInfo(System.IO.Path.Combine(System.IO.Path.Combine(x.destination, destinationDir.Name)))
//                        let destinationFile = new System.IO.FileInfo(System.IO.Path.Combine(destinationFilePath.FullName, f.Name))
//                        checkDir destinationFilePath                                         
//                        destinationFile
//                    let saveAndCompressImage (f:System.IO.FileInfo) =
//                        async { 
//                                try         
//                                    use! acquire = x.asyncGate.Acquire()     
//                                    use i = Image.FromFile(f.FullName)
//                                    use t = new Bitmap(i.Width,i.Height);
//                                    use g = Graphics.FromImage(t)
//                                    g.DrawImage(i,0,0)
//                                    i.Dispose()
//                                    g.Dispose()
//                                    let newFile = checkFile(f)                                
//                                    t.Save(newFile.FullName,iciJpegCodec, epParameters)                    
//                                    t.Dispose()
//                                    x.RaiseEventOnGuiThread x.completed newFile.FullName        
//                                with
//                                |_ ->()
//                              }
//                    let rec loop() = async {                
//                        let! msg = inbox.Receive()
//                        do! saveAndCompressImage(msg)
//                        return! loop()
//                    }
//                    loop()
//                )
//
//        member x.Start(quality:System.Int64, useAgent:bool) =
//            let asyncGate = new AsyncGate.RequestGate(System.Environment.ProcessorCount)
//            let dirSource = System.IO.DirectoryInfo(x.source)
//            let dirDestination = System.IO.DirectoryInfo(x.destination)
//            let checkDir (dir:System.IO.DirectoryInfo) = 
//                if not dir.Exists then
//                    dir.Create()                    
//            let checkFile (f:System.IO.FileInfo) =  
//                let destinationDir = f.Directory
//                let destinationFilePath = System.IO.DirectoryInfo(System.IO.Path.Combine(System.IO.Path.Combine(x.destination, destinationDir.Name)))
//                let destinationFile = new System.IO.FileInfo(System.IO.Path.Combine(destinationFilePath.FullName, f.Name))
//                checkDir destinationFilePath               
//                destinationFile
//            checkDir dirDestination
//            let rec visitor dir =  
//                seq { 
//                        yield! System.IO.Directory.EnumerateFiles(dir, x.pattern) 
//                        for subdir in System.IO.Directory.EnumerateDirectories(dir) do yield! visitor subdir
//                    }
//            let epQuality = new EncoderParameter(System.Drawing.Imaging.Encoder.Quality, quality)
//            let epCompression = new EncoderParameter(System.Drawing.Imaging.Encoder.Compression, quality)
//            let iciCodes = ImageCodecInfo.GetImageEncoders()
//            let iciJpegCodec = iciCodes.First(fun (i:ImageCodecInfo) -> i.MimeType = "image/jpeg")
//            let epParameters = new EncoderParameters(1);
//            epParameters.Param.[0] <- epQuality
//            let saveAndCompressImage (f:System.IO.FileInfo) =
//                async {          
//                        use! acquire = x.asyncGate.Acquire()                     
//                        use newImage = Image.FromFile(f.FullName)
//                        use tempImage = new Bitmap(newImage.Width, newImage.Height)
//                        use g=Graphics.FromImage(tempImage)
//                        do g.DrawImage(newImage,0,0)
//                        g.Dispose()
//                        newImage.Dispose()
//                        let newFile = checkFile(f)                                
//                        tempImage.Save(newFile.FullName,iciJpegCodec, epParameters)                    
//                        tempImage.Dispose()
//                        x.RaiseEventOnGuiThread x.completed newFile.FullName        
//                      }            
//            let files = visitor(dirSource.FullName)  
//            match useAgent with
//            | true ->   files      
//                        |> Seq.iter(fun f -> x.Agent.Post(new FileInfo(f)))
//            | false ->  files
//                        |> Seq.map (fun f -> saveAndCompressImage(new System.IO.FileInfo(f)))
//                        |> Async.Parallel            
//                        |> Async.RunSynchronously
//                        |> ignore
//            "Completed"
//    end