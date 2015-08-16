namespace Microsoft.WindowsAzure.StorageClient

open System.IO
open Microsoft.WindowsAzure
open Microsoft.WindowsAzure.StorageClient
open System.IO
open System.Diagnostics

[<AutoOpen>]
module AccountDetails = 
    let urlBase = "https://riscanet.blob.core.windows.net/stuff/"   
    let folder = "stuff"
    let account = "riscanet"
    let key = "ktT3brtjsL1mX+1uwfUKITK/SZfA3wEv1GpYClMep35b0i8nT3dRAQwpbkR9yQT4pRJ22Drghpgik9hAvjLBGQ=="
    let viewerProcess() = System.Diagnostics.Process.Start(@"E:\Ok\FSharpIntro\src\PhotoViewer\bin\Release\PhotoViewer.exe")

[<AutoOpen>]
module Extensions = 
    type CloudBlob with
        
        /// Asynchronously download blob contents as an array
        member blob.AsyncDownloadByteArray() = 
            async { 
                let ms = new MemoryStream(int blob.Properties.Length)
                do! Async.FromBeginEnd((fun (cb, s) -> blob.BeginDownloadToStream(ms, cb, s)), blob.EndDownloadToStream)
                return ms.ToArray() }
                        
        member blob.AsyncUploadFile(filePath) = 
            async { 
                use stream = File.OpenRead(filePath)
                do! Async.FromBeginEnd
                        ((fun (cb, s) -> blob.BeginUploadFromStream(stream, cb, s)), blob.EndUploadFromStream)
                stream.Close()
                return () }
            
        member blob.AsyncUploadByteArray(bytes : byte array) = 
            async { 
                use ms = new MemoryStream(bytes)
                do! Async.FromBeginEnd((fun (cb, s) -> blob.BeginUploadFromStream(ms, cb, s)), blob.EndUploadFromStream)
                ms.Close()
                return ms.ToArray() }

        member blob.AsyncDelete() = async {
                return! Async.FromBeginEnd((fun (cb, s) -> blob.BeginDeleteIfExists(cb, s)), blob.EndDeleteIfExists) }            

module AzureBlobHelpers = 
    type BlobTypes = 
        | CloudBlockBlob of CloudBlockBlob list
        | CloudPageBlob of CloudPageBlob list
        | CloudBlobDirectory of CloudBlobDirectory list
    
    let getBlobContainer = 
        let acct = 
            CloudStorageAccount.Parse(sprintf "DefaultEndpointsProtocol=https;AccountName=%s;AccountKey=%s" account key)
        let storage = acct.CreateCloudBlobClient()
        let container = storage.GetContainerReference(folder)
        let _ = container.CreateIfNotExist()
        container
    
    let rec getFiles folderPath = 
        seq { for file in Directory.EnumerateFiles(folderPath, "*.jpg") do yield file
              for dir in Directory.EnumerateDirectories(folderPath) do yield! getFiles dir }
        
    
    let sourceFolder = @"C:\temp\photos"
    
    //      ASYNCHRONOUS UPLOAD
    let upload sourceFolder (container : CloudBlobContainer) = 
        let fileImages = getFiles sourceFolder
        fileImages
        |> Seq.map(fun file -> (container.GetBlobReference(Path.GetFileName(file)), file))
        |> Seq.map (fun (blob, file) -> blob.AsyncUploadFile(file))
        |> Async.Parallel
        |> Async.Ignore
        |> Async.StartImmediate
    
    //upload sourceFolder getBlobContainer

    let listBlobs (container : CloudBlobContainer) = 
        let blockBlobs = 
            container.ListBlobs()
            |> Seq.filter (fun b -> b.GetType() = typeof<CloudBlockBlob>)
            |> Seq.map (fun b -> b :?> CloudBlockBlob)
            |> Seq.toList
        
        let pageBlobls = 
            container.ListBlobs()
            |> Seq.filter (fun b -> b.GetType() = typeof<CloudPageBlob>)
            |> Seq.map (fun b -> b :?> CloudPageBlob)
            |> Seq.toList
        
        let blobDirectory = 
            container.ListBlobs()
            |> Seq.filter (fun b -> b.GetType() = typeof<CloudBlobDirectory>)
            |> Seq.map (fun b -> b :?> CloudBlobDirectory)
            |> Seq.toList
        
        (CloudBlockBlob(blockBlobs), CloudPageBlob(pageBlobls), CloudBlobDirectory(blobDirectory))

    let deleteBlob (blob : CloudBlob) = 
        blob.AsyncDelete()
        