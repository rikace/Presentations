#I @"../../StartingWithFSharp/Lib"
#r "Microsoft.WindowsAzure.Diagnostics.dll"
#r "Microsoft.WindowsAzure.Diagnostics.StorageUtility.dll"
#r "Microsoft.WindowsAzure.StorageClient.dll"
#load "Common\\AzureExtensions.fs"

open System
open Microsoft.FSharp.Control
open System.IO
open Microsoft.WindowsAzure
open Microsoft.WindowsAzure.StorageClient

type Agent<'T> = MailboxProcessor<'T>

module ``Agent Basic`` =
  
    let cancellationToken = new System.Threading.CancellationTokenSource()
 
    let oneAgent =
           Agent.Start(fun inbox ->
             async { while true do
                       let! msg = inbox.Receive()
                       printfn "got message '%s'" msg } )
 
    oneAgent.Post "hi"


    let start<'T> (work : 'T -> unit) =
        Agent<obj>.Start(fun mb ->
            let rec loop () = async {

                    let! msg = mb.Receive()
                    match msg with
                    | :? 'T as msg' -> work msg'
                    | _ -> () // oops... undefined behaviour

                    return! loop () }
            loop () )

    let printInt = start<int>(fun value -> printfn "Print: %d" value)
    let printString = start<string>(fun value -> printfn "Print: %s" value)

    printInt.Post(7)
    printString.Post("Hello")

    printInt.Post("Hello")
    printString.Post(7)



    // 100k agents
    let alloftheagents =
            [ for i in 0 .. 100000 ->
               Agent.Start(fun inbox ->
                 async { while true do
                           let! msg = inbox.Receive()
                           if i % 10000 = 0 then
                               printfn "agent %d got message '%s'" i msg })]
 
    for agent in alloftheagents do
        agent.Post "ping!"


    (*
    In the echoAgent example, we used the Receive method to get messages from the underlying queue. 
    In many cases, Receive is appropriate, but it makes it difficult to filter messages because it removes them from the queue. 
    To selectively process messages, you might consider using the Scan method instead.
    Scanning for messages follows a different pattern than receiving them directly. Rather than processing the messages inline and always returning an asynchronous workflow,
    the Scan method accepts a filtering function that accepts a message and returns an Async<'T> option. 
    In other words, when the message is something you want to process, you return Some<Async<'T>; otherwise, you return None.
    *)

    type MessageScan = 
        | Message of obj

    let echoAgent =
      Agent<MessageScan>.Start(fun inbox ->
        let rec loop () =
          inbox.Scan(fun (Message(x)) ->
           match x with
           | :? string
           | :? int ->
             Some (async { printfn "%O" x
                           return! loop() })
           | _ -> printfn "<not handled>"; None)
        loop())

    Message "nuqneH" |> echoAgent.Post
    Message 123 |> echoAgent.Post
    Message [ 1; 2; 3 ] |> echoAgent.Post // not handled

    (* Scanning for messages does offer some flexibility with how you process messages, 
       but you need to be mindful of what you’re posting to the agent because messages not processed by Scan remain in the queue. 
       As the queue size increases, scans will take longer to complete, so you can quickly run into performance issues using this approach if you’re not careful *)


module ``Multi Agent Azure`` =

    let imageFolderDestination = @"C:\Temp\ImagesProcessed\";

    // http://www.teknology360.com/photos/bugghina1.jpg

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
        let fileName = imageFolderDestination + blob.Uri.Segments.[blob.Uri.Segments.Length-1]
        use outStream =  File.OpenWrite(fileName)
        do outStream.Write(pixels, 0, pixels.Length)
        fileName

    let downloadAll() = 
        for blob in container.ListBlobs() do 
            let name = downloadImage(container.GetBlobReference(blob.Uri.ToString()))
            printfn "Downloaded %s" name// (Path.GetFileNameWithoutExtension(name))
            ()

    // ***********************************************
    //      ASYNCHRONOUS 1 SEQ
    // ***********************************************
    let downloadImageAsync(blob : CloudBlob) = async {
        let! pixels = blob.AsyncDownloadByteArray()
        let fileName = imageFolderDestination + blob.Uri.Segments.[blob.Uri.Segments.Length-1]
        use outStream =  File.OpenWrite(fileName)
        do! outStream.AsyncWrite(pixels, 0, pixels.Length)
        return fileName }

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
    //      ASYNCHRONOUS MULTI AGENTs
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
