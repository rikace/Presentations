namespace Easj360FSharp


module AsyncWorker =


    open System
    open System.Threading
    open System.IO
    open Microsoft.FSharp.Control.WebExtensions

    type AsyncWorkerJobs<'T>(jobs) = 

        // Capture the synchronization context to allow us to raise events back on the GUI thread
        let syncContext = System.Threading.SynchronizationContext.Current

        // Check that we are being called from a GUI thread
        do match syncContext with 
            | null -> failwith "Failed to capture the synchronization context of the calling thread. The System.Threading.SynchronizationContext.Current of the calling thread is null"
            | _ -> ()


        let allCompleted  = new Event<unit>()
        let error         = new Event<System.Exception>()
        let canceled      = new Event<System.OperationCanceledException>()
        let jobCompleted  = new Event<int * 'T>()


        let asyncGroup = new CancellationTokenSource() 

        let raiseEventOnGuiThread (event:Event<_>) args =
            syncContext.Post(SendOrPostCallback(fun _ -> event.Trigger args),state=null)

        member x.Start()    = 
                                                       
            // Mark up the jobs with numbers
            let jobs = jobs |> List.mapi (fun i job -> (job,i+1))

            let work =  
                Async.Parallel 
                   [ for (job,jobNumber) in jobs do
                        yield 
                           async { let! result = job
                                   raiseEventOnGuiThread jobCompleted (jobNumber,result) } ]
                 |> Async.Ignore

            Async.StartWithContinuations
                ( work,
                  (fun res -> raiseEventOnGuiThread allCompleted res),
                  (fun exn -> raiseEventOnGuiThread error exn),
                  (fun exn -> raiseEventOnGuiThread canceled exn ),
                  asyncGroup.Token)

        member x.CancelAsync(message) = 
            asyncGroup.Cancel(); 
        
        member x.JobCompleted  = jobCompleted.Publish
        member x.AllCompleted  = allCompleted.Publish
        member x.Canceled   = canceled.Publish
        member x.Error      = error.Publish


    type AsyncWorker<'T>(p : Async<'T>,?cancellationToken) = 

            let cts =
                match cancellationToken with
                | None -> new CancellationTokenSource()
                | Some token ->
                      let cts = new CancellationTokenSource()
                      CancellationTokenSource.CreateLinkedTokenSource(token,cts.Token)

            let mutable syncContext : SynchronizationContext = null

            // A standard helper to raise an event on the GUI thread

            let raiseEventOnGuiThread (event:Event<_>) args =
                syncContext.Post((fun _ -> event.Trigger args),state=null)

            // Trigger one of the following events when the iteration completes.
            let completed = new Event<'T>()
            let error     = new Event<_>()
            let canceled   = new Event<_>()
            let progress  = new Event<int>()

            let doWork() = 
                Async.StartWithContinuations
                    ( p, 
                      (fun res -> raiseEventOnGuiThread completed res),
                      (fun exn -> raiseEventOnGuiThread error exn),
                      (fun exn -> raiseEventOnGuiThread canceled exn ),cts.Token)
                                
            member x.ReportProgress(progressPercentage) = 
                raiseEventOnGuiThread progress progressPercentage
        
            member x.RunAsync()    = 
                match syncContext with 
                | null -> ()
                | _ -> invalidOp "The operation is already in progress. RunAsync can't be called twice"

                syncContext <- 
                    match SynchronizationContext.Current with 
                    | null -> new SynchronizationContext()
                    | ctxt -> ctxt

                ThreadPool.QueueUserWorkItem(fun args -> doWork())

            member x.CancelAsync(?message:string) = 
                cts.Cancel()

            member x.ProgressChanged     = progress.Publish
            member x.Completed  = completed.Publish
            member x.Canceled   = canceled.Publish
            member x.Error      = error.Publish



    //------------------------------------------
    // Test code



    module SimpleTest = 
        let worker = new AsyncWorkerJobs<_>( [ for i in 0 .. 10 -> async { return i*i } ] )

        worker.JobCompleted.Add(fun (jobNumber, result) -> printfn "job %d completed with result %A" jobNumber result)
        worker.AllCompleted.Add(fun () -> printfn "all done!" )

        worker.Start()

    module WebTest = 

        open System.Net
        /// Fetch the contents of a web page, asynchronously
        let httpAsync(url:string) = 
            async { let req = WebRequest.Create(url) 
                
                    let! resp = req.AsyncGetResponse()
                
                    // rest is a callback
                    use stream = resp.GetResponseStream() 
                
                    use reader = new StreamReader(stream) 
                
                    let text = reader.ReadToEnd() 
                
                    return text }

        let urls = 
            [ "http://www.live.com"; 
              "http://news.live.com"; 
              "http://www.yahoo.com"; 
              "http://news.yahoo.com"; 
              "http://www.google.com"; 
              "http://news.google.com"; ] 

        let jobs =  [ for url in urls -> httpAsync url ]
    
        let worker = new AsyncWorkerJobs<_>(jobs)
        worker.JobCompleted.Add(fun (jobNumber, result) -> printfn "job %d completed with result %A" jobNumber result)
        worker.AllCompleted.Add(fun () -> printfn "all done!" )
        worker.Start()


    open System.Threading

    let sumArray (arr : int[]) =
        // Define a location in shared memory for counting
        let total = ref 0

        let half = arr.Length/2
        Async.Parallel [ for (a,b) in [(0,half-1);(half,arr.Length-1)]  do 
                            yield async { let _ = for i = a to b do
                                                      total := arr.[i] + !total
                                          return () } ]
          |> Async.Ignore
          |> Async.RunSynchronously
        !total

    sumArray [| 1;2;3 |]    
    sumArray [| 1;2;3;4 |]    

    let sumArray2 (arr : int[]) =

        let half = arr.Length/2
        Async.Parallel [ for (a,b) in [(0,half-1);(half,arr.Length-1)]  do 
                            yield async { let total = ref 0
                                          let _ = for i = a to b do
                                                      total := arr.[i] + !total
                                          return !total } ]
          |> Async.RunSynchronously
          |> Array.sum

    sumArray2 [| 1;2;3 |]    
    sumArray2 [| 1;2;3;4 |]    
