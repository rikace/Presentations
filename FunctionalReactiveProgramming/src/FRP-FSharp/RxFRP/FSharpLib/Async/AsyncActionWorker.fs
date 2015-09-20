namespace Easj360FSharp


module AsyncActionWorker =


    open System
    open System.Threading
    open System.IO
    open Microsoft.FSharp.Control.WebExtensions
    
    type JobStatus =
    | Queue
    | Schedule
    | InProgress
    | Completed
    | Faulted
    | Cancelled

    type EventActionCompleted(index:int) =
        inherit EventArgs()    
        member x.Index 
            with get () = index
    
    type EventFuncCompleted<'T>(index:int, result:'T) =
        inherit EventActionCompleted(index)
        member x.Result 
            with get () = result

    type EventJobStatus(index:int, status:JobStatus) =
        inherit EventArgs()
         member x.Index 
            with get () = index
        member x.Status 
            with get () = status

    type AsyncActionWorker<'T>() = 

        let syncContext = System.Threading.SynchronizationContext.Current

//        do match syncContext with 
//            | null -> failwith "Failed to capture the synchronization context of the calling thread. The System.Threading.SynchronizationContext.Current of the calling thread is null"
//            | _ -> ()
        
        let allCompleted  = new Event<unit>()
        let error         = new Event<System.Exception>()
        let canceled      = new Event<System.OperationCanceledException>()
        let jobFunCompleted  = new Event<EventFuncCompleted<'T>>()
        let jobActionCompleted = new Event<EventActionCompleted>()
        let jobStatus = new Event<EventJobStatus>()

        let asyncGroup = new CancellationTokenSource() 

        let raiseEventOnGuiThread (event:Event<_>) args =
            match syncContext with
            | null -> event.Trigger(args)
            | _ -> syncContext.Post(SendOrPostCallback(fun _ -> event.Trigger args),state=null)

        member private x.AsyncFuncHelper<'T>(action:Func<'T>) = 
            Async.FromBeginEnd(action.BeginInvoke, action.EndInvoke)

        member private x.AsyncActionHelper(action:Action) =
            Async.FromBeginEnd(action.BeginInvoke, action.EndInvoke)

        member x.StartFunc(jobs:Func<'T> list) = 
            let jobs = jobs |> List.mapi (fun i job -> (job,i+1))
            let work =  
                Async.Parallel 
                   [ for (job,jobNumber) in jobs do
                        yield 
                           async { 
                                   let! result = x.AsyncFuncHelper(job)
                                   raiseEventOnGuiThread jobFunCompleted (new EventFuncCompleted<'T>(jobNumber, result)) } ]                                   
                 |> Async.Ignore

            Async.StartWithContinuations
                ( work,
                  (fun res -> raiseEventOnGuiThread allCompleted res),
                  (fun exn -> raiseEventOnGuiThread error exn),
                  (fun exn -> raiseEventOnGuiThread canceled exn ),
                  asyncGroup.Token)

        member x.StartAction(jobs:Action list) = 
            let jobs = jobs |> List.mapi (fun i job -> (job,i+1))
            let work =  
                Async.Parallel 
                   [ for (job,jobNumber) in jobs do
                        yield 
                           async { 
                                   do! x.AsyncActionHelper(job)
                                   raiseEventOnGuiThread jobActionCompleted (new EventActionCompleted(jobNumber)) } ]
                 |> Async.Ignore

            Async.StartWithContinuations
                ( work,
                  (fun res -> raiseEventOnGuiThread allCompleted res),
                  (fun exn -> raiseEventOnGuiThread error exn),
                  (fun exn -> raiseEventOnGuiThread canceled exn ),
                  asyncGroup.Token)
    
        member x.parallelWorker n f = 
            MailboxProcessor.Start(fun inbox ->
                let workers = 
                    Array.init n (fun i -> MailboxProcessor.Start(f))
                let rec loop (i, index) = async {
                    let! msg = inbox.Receive()
                    workers.[i].Post((msg, index))
                    return! loop ((i+1) % n, index + 1)
                }
                loop (0, 0)
            )
    
        member x.agent = 
            x.parallelWorker Environment.ProcessorCount (fun inbox ->
                let rec loop() = async {            
                    //raiseEventOnGuiThread jobStatus (new EventJobStatus((snd msg), JobStatus.Schedule))    
                    let! msg = inbox.Receive()
                    do! x.AsyncActionHelper(fst msg)
                    raiseEventOnGuiThread jobActionCompleted ( new EventActionCompleted(snd msg) )
                    //raiseEventOnGuiThread jobStatus (new EventJobStatus((snd msg), JobStatus.Completed))
                    return! loop()
                }
                loop()
            )   

        member x.AddWork(job:Action) =
             x.agent.Post(job)
             //raiseEventOnGuiThread jobStatus (new EventJobStatus( (x.agent.CurrentQueueLength - 1), JobStatus.Queue))

        member x.CancelAsync(message) = 
            asyncGroup.Cancel(); 
    
        [<CLIEventAttribute>]    
        member x.JobFuncCompleted  = jobFunCompleted.Publish
        [<CLIEventAttribute>]    
        member x.JobActionCompleted  = jobActionCompleted.Publish
        [<CLIEventAttribute>]
        member x.AllCompleted  = allCompleted.Publish
        [<CLIEventAttribute>]
        member x.Canceled   = canceled.Publish
        [<CLIEventAttribute>]
        member x.Error      = error.Publish


        (*module SimpleTest = 
        let worker = new AsyncWorker<_>( [ for i in 0 .. 10 -> async { return i*i } ] )

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
    
        let worker = new AsyncWorker<_>(jobs)
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
    sumArray2 [| 1;2;3;4 |]   *)