#r "FSharp.PowerPack.dll"
#load "..\Utilities\show-wpf40.fsx"
open System
open System.IO
open System.Threading
open System.Net
open ShowWpf


type AsyncWorker<'T>(jobs: seq<Async<'T>>) =  

    // Capture the synchronization context to allow us 
    // to raise events back on the GUI thread
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
        let jobs = jobs |> Seq.mapi (fun i job -> (job,i+1))

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


module SimpleTest = 
    let worker = new AsyncWorker<_>( [ for i in 0 .. 10 -> async { return i*i } ] )

    worker.JobCompleted.Add(fun (jobNumber, result) -> showA (sprintf "job %d completed with result %A" jobNumber result))
    worker.AllCompleted.Add(fun () -> showA (sprintf "all done!" ))

    worker.Start()

module WebTest = 
    let httpAsync(url:string) = 
        async { let req = WebRequest.Create(url)                 
                let! resp = req.AsyncGetResponse()
                use stream = resp.GetResponseStream() 
                use reader = new StreamReader(stream) 
                return! reader.AsyncReadToEnd() }

    let urls = 
        [ "http://www.live.com"; 
          "http://news.live.com"; 
          "http://www.yahoo.com"; 
          "http://news.yahoo.com"; 
          "http://www.google.com"; 
          "http://news.google.com"; ] 

    let jobs =  [ for url in urls -> httpAsync url ]
    
    let worker = new AsyncWorker<_>(jobs)
    //worker.JobCompleted.Add(fun (jobNumber, result) -> printfn "job %d completed with result %A" jobNumber result)
    worker.JobCompleted.Add(fun (jobNumber, result) -> showA (sprintf "Job %d len %d" jobNumber result.Length))
    worker.AllCompleted.Add(fun () -> printfn "all done!" )
    worker.Start()
