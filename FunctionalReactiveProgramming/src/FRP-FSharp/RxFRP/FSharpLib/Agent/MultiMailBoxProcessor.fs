namespace Eaasj360

module ParallelWorker =

    // A reusable parallel worker model built on F# agents
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

let agent f = 
        parallelWorker 8 (fun inbox ->
            let rec loop() = async {                
                let! msg = inbox.Receive()
//                let blob = container.GetBlobReference(msg)
//                let! fileName = downloadImageAsync(blob)
                return! loop()

            }
            loop()
        )
        
        
//let agent f =
//    parallelWorker System.Environment.ProcessorCount (fun inbox ->
//        let rec loop() = async {
//        
//            let! msg = inbox.Receive()
//            
//            //let work = some Work
//            f(msg)
//            let work = msg
//            //let work = some Work
//            //let! workAsynx = some Async Work
//            
//            return! loop()
//        }
//        loop()
//    )          
//    

(*
// Parallel agent-based image downloader
// Agents (MailboxProcessor) provide building-block for other primitives - like parallelWorker 
// Re-introduces parallelism

type AzureImageDownloader(folder) = 
    let acct = CloudStorageAccount.Parse("DefaultEndpointsProtocol=https;AccountName=<YOUR_ACCOUNT_NAME_HERE>;AccountKey=<YOUR_ACCOUNT_KEY_HERE>")

    let storage = acct.CreateCloudBlobClient();
    let container = storage.GetContainerReference(folder);
    let _ = container.CreateIfNotExist()

    let downloadImageAsync(blob : CloudBlob) =
      async {
        let! pixels = blob.AsyncDownloadByteArray()
        let fileName = "thumbs-" + blob.Uri.Segments.[blob.Uri.Segments.Length-1]
        use outStream =  File.OpenWrite(fileName)
        do! outStream.AsyncWrite(pixels, 0, pixels.Length)
        return fileName
      }

    // A reusable parallel worker model built on F# agents
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
        parallelWorker 8 (fun inbox ->
            let rec loop() = async {
                
                let! msg = inbox.Receive()
                let blob = container.GetBlobReference(msg)
                let! fileName = downloadImageAsync(blob)
                return! loop()

            }
            loop()
        )

    member this.DownloadAll() = 
      for blob in container.ListBlobs() do
        agent.Post(blob.Uri.ToString())*)
