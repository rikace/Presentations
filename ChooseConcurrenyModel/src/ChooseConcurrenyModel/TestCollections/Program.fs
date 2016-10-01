open System
open System.IO
open CommonHelpers

module test = 
    let (|FileExtension|) path = Path.GetExtension path
    let (|FileName|) path = Path.GetFileNameWithoutExtension path

    let determinate path =
        match path with
        | FileExtension (".jpg" | ".png" | ".gif") -> "image"
        | FileName "test" -> "test"
        | _ -> "NOS" 


    determinate "photo.jpg"

    let dateTime = DateTime.Now
    let d1, t1 =  dateTime.Date, dateTime.TimeOfDay

    let (|Date|) (d:DateTime) = d.Date
    let (|Time|) (d:DateTime) = d.TimeOfDay
    let (|Hour|) (d:DateTime) = d.TimeOfDay.Hours

    let myFunDate (Date d & Time t & Hour h) = printfn "Date:%A Time:%A Hour:%A" d t h

    myFunDate dateTime

module Async =
      open System.Threading.Tasks
     
      let StartDisposable(op:Async<unit>) =
          let ct = new System.Threading.CancellationTokenSource()
          Async.Start(op, ct.Token)
          { new IDisposable with 
              member x.Dispose() = ct.Cancel() }

      let map f a = async.Bind(a, f >> async.Return)

      let chooseTasks (a:Task<'T>) (b:Task<'U>) : Async<Choice<'T * Task<'U>, 'U * Task<'T>>> =
        async { 
            let! ct = Async.CancellationToken
            let i = Task.WaitAny( [| (a :> Task);(b :> Task) |],ct)
            if i = 0 then return (Choice1Of2 (a.Result, b))
            elif i = 1 then return (Choice2Of2 (b.Result, a)) 
            else return! failwith (sprintf "unreachable, i = %d" i) }


  type RequestGate(n:int) =
        let semaphore = new System.Threading.Semaphore(n,n)
        member x.Aquire(?timeout) = 
            async { let! ok = Async.AwaitWaitHandle(semaphore, ?millisecondsTimeout=timeout)
                    if ok then return { new System.IDisposable with
                                            member x.Dispose() =
                                                semaphore.Release() |> ignore }
                    else return! failwith "Handle not aquired" }


    let inline ParallelWithThrottle limit f items =
        let semaphore = new System.Threading.Semaphore(limit, limit)
        let ops = items |> Seq.map(fun i ->    async {
                let! ok = Async.AwaitWaitHandle(semaphore)
                try
                    let! result = f i
                    return result
                finally semaphore.Release() |> ignore })
        ops |> Async.Parallel 
        

module AgentWithThrotle =
    open System.Collections.Concurrent

    type JobRequest<'T> =
        {
            Id : int
            WorkItem : 'T
        }

    type WorkRequest<'T> =
        | Job of JobRequest<'T>
        | End

    let inline doParallelWithThrottle<'a, 'b> limit f items =
        let itemArray = Seq.toArray items
        let itemCount = Array.length itemArray
        let resultMap = ConcurrentDictionary<int, 'b>()
        use block = new BlockingCollection<WorkRequest<'a>>(1)
        use completeBlock = new BlockingCollection<unit>(1)
        let monitor =
            MailboxProcessor.Start(fun inbox ->
                let rec inner complete =
                    async {
                        do! inbox.Receive()
                        if complete + 1 = limit then
                            completeBlock.Add(())
                            return ()
                        else
                            return! inner <| complete + 1
                    }
                inner 0)
        let createAgent () =
            MailboxProcessor.Start(
                fun inbox ->
                    let rec inner () = async {
                            let! request = async { return block.Take() }
                            match request with
                            | Job job ->
                                let! result = async { return f (job.WorkItem) }
                                resultMap.AddOrUpdate(job.Id, result, fun _ _ -> result) |> ignore
                                return! inner ()
                            | End  ->
                                monitor.Post ()
                                return ()
                        }
                    inner ()
            )
        let agents =
            [| for i in 1..limit -> createAgent() |]
        itemArray
        |> Array.mapi (fun i item -> Job { Id = i; WorkItem = item })
        |> Array.iter (block.Add)

        [1..limit]
        |> Seq.iter (fun x -> block.Add(End))

        completeBlock.Take()
        let results = Array.zeroCreate itemCount
        resultMap
        |> Seq.iter (fun kv -> results.[kv.Key] <- kv.Value)
        results                            




open System
open System.Threading
open System.Threading.Tasks

[<AutoOpen>]
module AsyncEx =

    type Async with

          /// Creates an async computation which runs the provided sequence of computations and completes
          /// when all computations in the sequence complete. Up to parallelism computations will
          /// be in-flight at any given point in time. Error or cancellation of any computation in
          /// the sequence causes the resulting computation to error or cancel, respectively.
          static member ParallelIgnore (parallelism:int) (xs:seq<Async<_>>) = async {                
            let sm = new SemaphoreSlim(parallelism)
            let cde = new CountdownEvent(1)
            let tcs = new TaskCompletionSource<unit>()
            let inline tryComplete () =
              if cde.Signal() then
                tcs.SetResult(())
            let inline ok _ =
              sm.Release() |> ignore
              tryComplete ()                
            let inline err (ex:exn) =
              tcs.TrySetException ex |> ignore
              sm.Release() |> ignore                        
            let inline cnc (ex:OperationCanceledException) =      
              tcs.TrySetCanceled() |> ignore
              sm.Release() |> ignore
            try
              use en = xs.GetEnumerator()
              while not (tcs.Task.IsCompleted) && en.MoveNext() do
                sm.Wait()
                cde.AddCount(1)
                Async.StartWithContinuations(en.Current, ok, err, cnc)                              
              tryComplete ()
              do! tcs.Task |> Async.AwaitTask
            finally      
              cde.Dispose()    
              sm.Dispose() }




    let run x = Async.RunSynchronously x

    let runParallel (workflow : Async<'T>) =
        [| for i in 1 .. 100 -> workflow |] 
        |> Async.ParallelIgnore 10
        |> run

    let a = async { return 42 }
    let b = async { return run (async { return 42 })}

    runParallel a // Real: 00:00:00.031, CPU: 00:00:00.015, GC gen0: 0, gen1: 0, gen2: 0
    runParallel b // Real: 00:00:00.023, CPU: 00:00:00.031, GC gen0: 0, gen1: 0, gen2: 0



[<EntryPoint>]
let main argv = 


//    Lists.ArrayTest.start()
//    printfn "Complete Array"
//    Console.ReadLine() |> ignore
//    Lists.ListTest.start()
//    printfn "Complete List"
//    Console.ReadLine() |> ignore
//    Lists.ResizeArrayTest.start()
//    printfn "Complete Resize Array"
//    Console.ReadLine() |> ignore
//    Lists.SeqTest.start()
//    printfn "Complete Seq"
//    Console.ReadLine() |> ignore
//    Lists.SetTest.start()
//    printfn "Complete Set"
//    Console.ReadLine() |> ignore
//    Lists.HashTest.start()


    Lists.MapTest.start()
    Lists.HashMapTest.start()
    Lists.DicTest.start()

//    let words = ParallelizingFuzzyMatch.Data.Words
//
//    
//    BenchPerformance.Time("Parallel Init", fun () -> 
//        
//        let wordwordTuple = Array.Parallel.init words.Length (fun i -> (i, words.[i]))
//        let test = wordwordTuple 
//        ())
//
//    BenchPerformance.Time("Init",fun () -> 
//        let wordwordTuple' = Array.init words.Length (fun i -> (i, words.[i]))
//        let t = wordwordTuple'
//        ())


    // insert
    // remove
    // lookup
    // sort 


//    let pathFileText = @"..\..\..\Data\Shakespeare"
//
//    let text =  File.ReadAllText(pathFileText)

    Console.ReadLine() |> ignore
    0 // return an integer exit code
