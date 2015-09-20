namespace Easj360FSharp

open System
open System.IO
open System.Drawing
open System.Drawing.Imaging
open System.Xml
open System.Xml.Linq

module ImageResizer = 

    type FilePaths = {  ImagePath:string;
                        CompressedImagePath:string;
                        ThumbnailPath:string }

    type MultiImageResizer() =
        
        let evMessage = new Event<string>()

        let checkAndCreateDir d = if not(Directory.Exists(d)) then Directory.CreateDirectory(d) |> ignore

        let createImage(originalImage:string, filesPaths:FilePaths, nameNewImage:string,jpegCodec:ImageCodecInfo, quality:int64) = async {        
            let imagePathUnchaged = Path.Combine(filesPaths.ImagePath,nameNewImage)
            let imagePathThumbnail = Path.Combine(filesPaths.ThumbnailPath,nameNewImage)
            let imagePatCompressed = Path.Combine(filesPaths.CompressedImagePath,nameNewImage)

            checkAndCreateDir(filesPaths.ImagePath)
            checkAndCreateDir(filesPaths.CompressedImagePath)
            checkAndCreateDir(filesPaths.ThumbnailPath)

            File.Copy(originalImage, imagePathUnchaged, true)
            
            evMessage.Trigger (sprintf "Processing original file %s..." originalImage)

            use photoImg = Image.FromFile(originalImage)

            let size = new Size(640, 480)
            let sourceWidth = photoImg.Width
            let sourceHeight = photoImg.Height
            let nPercentW = (float(size.Width) / float(sourceWidth))
            let nPercentH = (float(size.Height) / float(sourceHeight))
            let nPercent =  if (nPercentH < nPercentW) then nPercentH
                            else nPercentW
            let destWidth = int(float(sourceWidth) * nPercent)
            let destHeight = int(float(sourceHeight) * nPercent)
            
            use bmp = new Bitmap(destWidth, destHeight)
            use gr = System.Drawing.Graphics.FromImage(bmp)                    
            gr.SmoothingMode <- System.Drawing.Drawing2D.SmoothingMode.HighQuality
            gr.CompositingQuality <- System.Drawing.Drawing2D.CompositingQuality.HighQuality
            gr.InterpolationMode <- System.Drawing.Drawing2D.InterpolationMode.HighQualityBicubic
            let rectDestination = new System.Drawing.Rectangle(0, 0, destWidth, destHeight)
            gr.DrawImage(photoImg, rectDestination, 0, 0, sourceWidth, sourceHeight, GraphicsUnit.Pixel)
            let qualityParam = new EncoderParameter(System.Drawing.Imaging.Encoder.Quality, quality)
            let encoderParams = new EncoderParameters(1)
            encoderParams.Param.[0] <- qualityParam
            evMessage.Trigger (sprintf "Processing compressed file %s..." imagePatCompressed)
            bmp.Save(imagePatCompressed, jpegCodec, encoderParams)
            
            let thumbWidth = 160
            let thumbHeight = Math.Max(((sourceHeight / sourceWidth) * thumbWidth), thumbWidth);
            use thumbBmp = new Bitmap(thumbWidth, thumbHeight)
            use grThumb = System.Drawing.Graphics.FromImage(thumbBmp)
            grThumb.SmoothingMode <- System.Drawing.Drawing2D.SmoothingMode.HighQuality
            grThumb.CompositingQuality <- System.Drawing.Drawing2D.CompositingQuality.HighQuality
            grThumb.InterpolationMode <- System.Drawing.Drawing2D.InterpolationMode.High
            let rectDestination = new System.Drawing.Rectangle(0, 0, thumbWidth, thumbHeight)
            grThumb.DrawImage(photoImg, rectDestination, 0, 0, sourceWidth, sourceHeight, GraphicsUnit.Pixel)

            evMessage.Trigger (sprintf "Processing thumbnail file %s..." imagePathThumbnail)
            thumbBmp.Save(imagePathThumbnail, System.Drawing.Imaging.ImageFormat.Jpeg)

            return (imagePathUnchaged, imagePathThumbnail, imagePatCompressed)  }

        let jpegCodec = 
            let codecs = ImageCodecInfo.GetImageEncoders()
            (codecs |> Array.filter( fun f -> f.MimeType = "image/jpeg")).[0]

        let getImageDirs dir = 
            let rec visitor dir filter=  
                        seq { yield! Directory.GetFiles(dir, filter) 
                              for subdir in Directory.GetDirectories(dir) do yield! visitor subdir filter}
            visitor dir "*.jpg"

//        let createXml dirImages = 
//            let elem = new XElement("data", new XAttribute("startalbumindex", 0),
//                         new XAttribute("transition", "NoTransition"));
//            
//
//            getImageDirs dirImages
//            |> Array.iter(fun dir ->  let mainImagesCurrentAlbum = Path.Combine(mainImages, dir.Name)
//                                      let mainThumbImages = Path.Combine(mainImagesCurrentAlbum, "ThumbnailImages")
//                                      let mainCompressImages = Path.Combine(mainImagesCurrentAlbum, "Images")
//                                      let index = ref 0
//                                      let album = new XElement("album", new XAttribute("title", dir.Name),
//                                                        new XAttribute("description", String.Format("Photo album {0}", dir.Name)),
//                                                        new XAttribute("source", String.Format("../Assets/{0}/{1}/{2}", dir.Name, "ThumbnailImages", "t_" + dir.Name + "_1.jpg")),
//                                                        new XAttribute("transition", "NoTransition"),
//                                                        dir.GetFiles("*.jpg")
//                                                        |> Array.iter(fun f -> 
//                                                            ignore(new XElement("slide", new XAttribute("title", dir.Name + "_" + (index).ToString()),
//                                                                new XAttribute("description", dir.Name + "_" + ((index := index + 1)).ToString()),
//                                                                new XAttribute("source", String.Format("../Assets/{0}/Images/{1}", dir.Name, Path.GetFileName(SaveJpeg(dir.Name, f.FullName, mainCompressImages, index, 145)))),
//                                                                new XAttribute("thumbnail", String.Format("../Assets/{0}/ThumbnailImages/{1}", dir.Name, Path.GetFileName(CreateThumbnail(dir.Name, f.FullName, mainThumbImages, index)))),
//                                                                new XAttribute("link", "")))))
//                                      elem.Add(album))
//
//            let doc = new XDocument();
//            doc.Add(elem);
//            //doc.Save(@"H:\AlbumImage\Images.xml");

// H:\Media\Immagini\Food

        [<CLIEventAttribute>]
        member x.MessageImage = evMessage.Publish

        member x.Start(dir:string) =
            let filesDir = {    ImagePath = @"L:\Destination\Original";
                                CompressedImagePath = @"L:\Destination\Compressed";
                                ThumbnailPath = @"L:\Destination\Thumbnail" }
            getImageDirs(dir)
            |> Seq.toArray
            |> Array.map (fun f -> createImage(f, filesDir, System.IO.Path.GetFileName(f) + ".jpg", jpegCodec, 130L)) 
            |> Async.Parallel
            |> Async.RunSynchronously




    type ImageResizerQueueAgent(sourceImages:string, destinationImages:string, quality:int64) =

        let queueLength= 8
        let token = new System.Threading.CancellationTokenSource()
        let copyImages = new BlockingQueueAgent<_>(queueLength)    
        let createThumbnailImages = new BlockingQueueAgent<_>(queueLength)    
        let compressImages = new BlockingQueueAgent<_>(queueLength)   

        [<LiteralAttribute>]
        let Thumbnail = "Thumbnail"
        [<LiteralAttribute>]
        let Compress = "Compress"
        [<LiteralAttribute>]
        let Original = "Original"

        let jpegCodec = 
            let codecs = ImageCodecInfo.GetImageEncoders()
            (codecs |> Array.filter( fun f -> f.MimeType = "image/jpeg")).[0]

        let checkAndCreateDir dir = 
            let tempDir = Path.Combine(destinationImages, dir)
            if not(Directory.Exists(tempDir)) then Directory.CreateDirectory(tempDir)|>ignore

        let copyImage source = 
            async{ File.Copy(source, Path.Combine(destinationImages, Original, Path.GetFileName(source)), true) }

        let createThumbnail source = 
            async {
                use photoImg = Image.FromFile(source)
                let sourceWidth = photoImg.Width
                let sourceHeight = photoImg.Height                
                let thumbWidth = 160
                let thumbHeight = Math.Max(((sourceHeight / sourceWidth) * thumbWidth), thumbWidth);
                use thumbBmp = new Bitmap(thumbWidth, thumbHeight)
                use grThumb = System.Drawing.Graphics.FromImage(thumbBmp)
                grThumb.SmoothingMode <- System.Drawing.Drawing2D.SmoothingMode.HighQuality
                grThumb.CompositingQuality <- System.Drawing.Drawing2D.CompositingQuality.HighQuality
                grThumb.InterpolationMode <- System.Drawing.Drawing2D.InterpolationMode.High
                let rectDestination = new System.Drawing.Rectangle(0, 0, thumbWidth, thumbHeight)
                grThumb.DrawImage(photoImg, rectDestination, 0, 0, sourceWidth, sourceHeight, GraphicsUnit.Pixel)
                thumbBmp.Save(Path.Combine(destinationImages, Thumbnail, Path.GetFileName(source)), System.Drawing.Imaging.ImageFormat.Jpeg)    }
        
        let compressImage source = 
            async {
                use photoImg = Image.FromFile(source)
                let size = new Size(640, 480)
                let sourceWidth = photoImg.Width
                let sourceHeight = photoImg.Height
                let nPercentW = (float(size.Width) / float(sourceWidth))
                let nPercentH = (float(size.Height) / float(sourceHeight))
                let nPercent =  if (nPercentH < nPercentW) then nPercentH
                                else nPercentW
                let destWidth = int(float(sourceWidth) * nPercent)
                let destHeight = int(float(sourceHeight) * nPercent)
            
                use bmp = new Bitmap(destWidth, destHeight)
                use gr = System.Drawing.Graphics.FromImage(bmp)                    
                gr.SmoothingMode <- System.Drawing.Drawing2D.SmoothingMode.HighQuality
                gr.CompositingQuality <- System.Drawing.Drawing2D.CompositingQuality.HighQuality
                gr.InterpolationMode <- System.Drawing.Drawing2D.InterpolationMode.HighQualityBicubic
                let rectDestination = new System.Drawing.Rectangle(0, 0, destWidth, destHeight)
                gr.DrawImage(photoImg, rectDestination, 0, 0, sourceWidth, sourceHeight, GraphicsUnit.Pixel)
                let qualityParam = new EncoderParameter(System.Drawing.Imaging.Encoder.Quality, quality)
                let encoderParams = new EncoderParameters(1)
                encoderParams.Param.[0] <- qualityParam
                bmp.Save(Path.Combine(destinationImages, Compress, Path.GetFileName(source)), jpegCodec, encoderParams) }

        let loadImagesProcess source = async {
                let rec visitor dir filter=  
                            seq { yield! Directory.GetFiles(dir, filter) 
                                  for subdir in Directory.GetDirectories(dir) do yield! visitor subdir filter}
                for file in visitor source "*.jpg" do
                    do! copyImages.AsyncAdd(file) }

        let copyImageProcess = async {
            while true do
                let! file = copyImages.AsyncGet()
                do! copyImage(file)
                do! createThumbnailImages.AsyncAdd(file)   }

        let creatThumbNailProcess = async {
            while true do
                let! file = createThumbnailImages.AsyncGet()
                do! createThumbnail(file)
                do! compressImages.AsyncAdd(file) }

        let compressIMageProcess = async {
            while true do
                let! file = compressImages.AsyncGet()
                do! compressImage(file) }

        member x.Start() = 
            checkAndCreateDir Original
            checkAndCreateDir Thumbnail
            checkAndCreateDir Compress
            Async.Start(loadImagesProcess(sourceImages), token.Token)
            Async.Start(copyImageProcess, token.Token)
            Async.Start(creatThumbNailProcess, token.Token)
            Async.Start(compressIMageProcess, token.Token)
            token
