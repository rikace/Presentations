#I @"C:\Git\Presentations\FS-Intro\src\StartingWithFSharp\Lib"
#r "Microsoft.WindowsAzure.Diagnostics.dll"
#r "Microsoft.WindowsAzure.Diagnostics.StorageUtility.dll"
#r "Microsoft.WindowsAzure.StorageClient.dll"
#load "Common\\AzureExtensions.fs"

open Microsoft.WindowsAzure
open Microsoft.WindowsAzure.StorageClient
open System.IO
open AccountDetails


let acct = CloudStorageAccount.Parse(sprintf "DefaultEndpointsProtocol=https;AccountName=%s;AccountKey=%s" account key)
let storage = acct.CreateCloudBlobClient();
let container = storage.GetContainerReference(folder);
let _ = container.CreateIfNotExist()

// start viewer
viewerProcess()



// ***********************************************
//      SYNCHRONOUS 
// ***********************************************
let downloadImage(blob : CloudBlob) =
    let pixels = blob.DownloadByteArray()
    let fileName = @"c:\temp\photos\thumbs-" + blob.Uri.Segments.[blob.Uri.Segments.Length-1]
    use outStream =  File.OpenWrite(fileName)
    do outStream.Write(pixels, 0, pixels.Length)
    fileName

let downloadAll() = 
    for blob in container.ListBlobs() do 
        let name = downloadImage(container.GetBlobReference(blob.Uri.ToString()))
        ()

let downloadImageAsync(blob : CloudBlob) = async {
    let! pixels = blob.AsyncDownloadByteArray()
    let fileName = @"c:\Temp\photos\thumbs-" + blob.Uri.Segments.[blob.Uri.Segments.Length-1]
    use outStream =  File.OpenWrite(fileName)
    do! outStream.AsyncWrite(pixels, 0, pixels.Length)
    return fileName }


// ***********************************************
//      ASYNCHRONOUS 1 SEQ
// ***********************************************
let downloadAllAsync() = 
    let cancelToken = new System.Threading.CancellationTokenSource()
    let comp = async { for blob in container.ListBlobs() do 
                         let! name = downloadImageAsync(container.GetBlobReference(blob.Uri.ToString()) ) 
                         () }
    Async.StartImmediate (comp, cancelToken.Token)
    cancelToken

let cancel = downloadAllAsync() 
cancel.Cancel()

// ***********************************************
//      ASYNCHRONOUS 2 PARALLEL
// ***********************************************
       
let downloadAllAsyncParallel() = 
    let cancelToken = new System.Threading.CancellationTokenSource()
    let comp = 
        container.ListBlobs()
        |> Seq.map (fun blob -> downloadImageAsync(container.GetBlobReference(blob.Uri.ToString())))
        |> Async.Parallel // what is it the problem here ?? 
        |> Async.Ignore
    Async.StartImmediate (comp, cancelToken.Token)
    cancelToken

let cancel' = downloadAllAsyncParallel()
cancel'.Cancel()

// ***********************************************
//      ASYNCHRONOUS 3 AGENT
// ***********************************************
let downloadAllAsyncAgent() =
    let cancelToken = new System.Threading.CancellationTokenSource()
    let agent = MailboxProcessor.Start((fun inbox ->
                    let rec loop() = async {
                        let! msg = inbox.Receive()
                        printfn "msg %s" msg
                        let blob = container.GetBlobReference(msg)
                        let! fileName = downloadImageAsync(blob)
                        return! loop() }
                    loop()), cancelToken.Token)

    for blob in container.ListBlobs() do
        printfn "%s" (blob.Uri.ToString())
        agent.Post(blob.Uri.ToString())
    cancelToken

let cancel'' = downloadAllAsyncAgent()
cancel''.Cancel()


// ***********************************************
//      ASYNCHRONOUS 4 MULTI AGENTs
// ***********************************************
let downloadAllMutiAgent() =
    let cancelToken = new System.Threading.CancellationTokenSource()

    let parallelWorker n f = 
        MailboxProcessor.Start((fun inbox ->
            let workers = Array.init n (fun i -> MailboxProcessor.Start(f))
            let rec loop i = async {                
                let! msg = inbox.Receive()
                workers.[i].Post(msg)
                return! loop ((i+1) % n) }
            loop 0 ), cancelToken.Token)

    let cpuCount = System.Environment.ProcessorCount

    let agent = 
        parallelWorker cpuCount (fun inbox ->
            let rec loop() = async {
                let! msg = inbox.Receive()
                let blob = container.GetBlobReference(msg)
                let! fileName = downloadImageAsync(blob)
                return! loop() }
            loop())

    for blob in container.ListBlobs() do
        agent.Post(blob.Uri.ToString())
    cancelToken
        
let cancel''' = downloadAllMutiAgent() 
cancel'''.Cancel()


