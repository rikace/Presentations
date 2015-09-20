namespace Easj360FSharp
#nowarn "40"

module AgentParallelWorker =

    open Microsoft.FSharp.Control
    open System.Collections.Concurrent
    open System
    open System.Threading
    open System.IO

    type Action with
        member x.AsyncAction() =
            Async.FromBeginEnd(x.BeginInvoke, x.EndInvoke)
    
    type Func<'a> with
        member x.AsyncFunc() =
            Async.FromBeginEnd(x.BeginInvoke, x.EndInvoke)

    type Message<'a> =
    | M of int * Action
    | F of int * Func<'a>
    | Stop
    | Reset
  
    type Status =
    | Schedule
    | Queue
    | Cancell
    | Processing
    | Completed
     
    type AgentPool<'T>(poolSize:int) =
        let syncContext = System.Threading.SynchronizationContext.Current 
//
//        do match syncContext with 
//            | null -> failwith "Failed to capture the synchronization context of the calling thread. The System.Threading.SynchronizationContext.Current of the calling thread is null"
//            | _ -> ()

        let allCompleted  = new Event<unit>()
        let error         = new Event<System.Exception * int>()
        let canceled      = new Event<System.OperationCanceledException>()
        let jobCompleted  = new Event<int * 'T>()
        let statusChanged = new Event<int * Status>()
        let poolStoped    = new Event<unit>()

//        let raiseEventOnGuiThread (event:Event<_>) args =
//            syncContext.Post(SendOrPostCallback(fun _ -> event.Trigger args),state=null)

        member private x.parallelWorker(n, f) =
                            MailboxProcessor.Start(fun inbox ->
                                let workers = 
                                    Array.init n (fun i -> MailboxProcessor.Start(f))
                                let rec loop(i, index) = async {                
                                    let! msg = inbox.Receive()
                                    try
                                        match msg with
                                            | Stop              ->  workers |> Array.iter( fun x -> x.Post(Stop))
                                                                   // raiseEventOnGuiThread poolStoped ()
                                                                    return ()
                                            | Reset             ->  return! loop(0,0)
                                            | M(index,action)   ->  workers.[i].Post(M(index,action))
                                            | F(index,func)     ->  workers.[i].Post(F(index,func))
                                        return! loop(((i+1) % n), index+1)
                                    with
                                    | ex -> //raiseEventOnGuiThread error (ex,index)
                                            return! loop( 0 , index)
                                }
                                loop (0,0)
                            )

        member private x.Agent = 
                    x.parallelWorker(poolSize, (fun inbox ->
                        let rec loop() = async {                
                            let! msg = inbox.Receive()
                            match msg with
                            | M(m,action) -> do! action.AsyncAction()
                                            // raiseEventOnGuiThread jobCompleted (m, Unchecked.defaultof<'T>)
                                             return! loop()
                            | F(m,func)   -> let! result = func.AsyncFunc()
                                             return! loop()
                            | Stop        -> //raiseEventOnGuiThread canceled
                                             return ()
                            | Reset       -> return! loop()
                        }
                        loop()
                    ))

        [<CLIEvent>]
        member x.JobCompleted   = jobCompleted.Publish
        [<CLIEvent>]
        member x.AllCompleted   = allCompleted.Publish
        [<CLIEvent>]
        member x.Canceled       = canceled.Publish
        [<CLIEvent>]
        member x.Error          = error.Publish
        [<CLIEvent>]
        member x.StatusChanged  = statusChanged.Publish
        [<CLIEvent>]
        member x.PoolStop       = poolStoped.Publish

        member x.Add(msg) =
               // raiseEventOnGuiThread statusChanged (0,Status.Schedule)
                x.Agent.Post(msg)

        member x.AddRange(msgs) =
                msgs|> Array.iter(fun m -> x.Agent.Post(m))
             //   raiseEventOnGuiThread allCompleted ()

