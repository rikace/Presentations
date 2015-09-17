namespace Easj360FSharp

open System.IO
open System
open System.Threading
open System.Collections.Generic

module CollectFiles = 
 
    type System.IO.Directory with 
    //        static member AsyncGetFiles(directory) = async { return System.IO.Directory.GetFiles(directory) } 
    //        static member AsyncGetDirectories(path) = async { return System.IO.Directory.GetDirectories(path) } 
            static member AsyncGetFiles(path:string, searchPattern:string, searchOption:SearchOption) = 
                            let fn = new Func<string * string * SearchOption, string[]>(Directory.GetFiles)    
                            Async.FromBeginEnd((path, searchPattern, searchOption), fn.BeginInvoke, fn.EndInvoke)
            static member AsyncGetDirectories(path:string, searchPattern:string, searchOption:SearchOption) = 
                            let fn = new Func<string * string * SearchOption, string[]>(Directory.GetDirectories)    
                            Async.FromBeginEnd((path, searchPattern, searchOption), fn.BeginInvoke, fn.EndInvoke)



    /// A convenience type alias for 'MailboxProcessor<T>' type
    type Agent<'T> = MailboxProcessor<'T>

    /// Type of messages internally used by 'BlockingQueueAgent<T>'
    type internal BlockingAgentMessage<'T> = 
      // Send item to the queue and block until it is added 
      | Add of 'T * AsyncReplyChannel<unit> 
      // Get item from the queue (block if no item available)
      | Get of AsyncReplyChannel<'T>

 
    /// <summary> Agent that implements an asynchronous blocking queue. </summary>
    /// <remarks>
    ///   The queue has maximal length (maxLength) and if the queue is 
    ///   full, adding to it will (asynchronously) block the caller. When
    ///   the queue is empty, the caller will be (asynchronously) blocked
    ///   unitl an item is available.
    /// </remarks>
    type BlockingQueueAgent<'T>(maxLength) =
      do 
        if maxLength <= 0 then 
          invalidArg "maxLenght" "Maximal length of the queue should be positive."

      // We keep the number of elements in the queue in a local field to
      // make it immediately available (so that 'Count' property doesn't 
      // have to use messages - which would be a bit slower)
      [<VolatileField>]
      let mutable count = 0

      let agent = Agent.Start(fun agent ->
        // Keeps a list of items that are in the queue
        let queue = new Queue<_>()
        // Keeps a list of blocked callers and additional values
        let pending = new Queue<_>()

        // If the queue is empty, we cannot handle 'Get' message
        let rec emptyQueue() = 
          agent.Scan(fun msg ->
            match msg with 
            | Add(value, reply) -> Some <| async {
                queue.Enqueue(value)
                count <- queue.Count
                reply.Reply()
                return! nonEmptyQueue() }
            | _ -> None )

        // If the queue is non-empty, we can handle all messages
        and nonEmptyQueue() = async {
          let! msg = agent.Receive()
          match msg with 
          | Add(value, reply) -> 
              // If the queue has space, we add item and notif caller back;
              // if it is full, we enqueue the request (and block caller)
              if queue.Count < maxLength then 
                queue.Enqueue(value)
                count <- queue.Count
                reply.Reply()
              else 
                pending.Enqueue(value, reply) 
              return! nonEmptyQueue()
          | Get(reply) -> 
              let item = queue.Dequeue()
              // We took item from the queue - check if there are any blocked callers
              // and values that were not added to the queue because it was full
              while queue.Count < maxLength && pending.Count > 0 do
                let itm, caller = pending.Dequeue()
                queue.Enqueue(itm)
                caller.Reply()
              count <- queue.Count
              reply.Reply(item)
              // If the queue is empty then switch the state, otherwise loop
              if queue.Count = 0 then return! emptyQueue()
              else return! nonEmptyQueue() }

        // Start with an empty queue
        emptyQueue() )

      /// Returns the number of items in the queue (immediately)
      /// (excluding items that are being added by callers that have been
      /// blocked because the queue was full)
      member x.Count = count

      /// Asynchronously adds item to the queue. The operation ends when
      /// there is a place for the item. If the queue is full, the operation
      /// will block until some items are removed.
      member x.AsyncAdd(v:'T, ?timeout) = 
        agent.PostAndAsyncReply((fun ch -> Add(v, ch)), ?timeout=timeout)

      /// Asynchronously gets item from the queue. If there are no items
      /// in the queue, the operation will block unitl items are added.
      member x.AsyncGet(?timeout) = 
        agent.PostAndAsyncReply(Get, ?timeout=timeout)

 
    let folders = new BlockingQueueAgent<option<string>>(100) 
    let files = new BlockingQueueAgent<option<string>>(100) 
        

    let folderCollector path = 
      async { let rec loop path =  
                async { do! folders.AsyncAdd(Some path) 
                        let! dirs = Directory.AsyncGetDirectories(path, "*.*", SearchOption.AllDirectories)
                        do! [ for z in dirs -> loop z ] |> Async.Parallel |> Async.Ignore }  
              do! loop path  
              do! folders.AsyncAdd(None) } 
 
    let fileCollector() = 
      async { let rec loop() =  
                async { let! dir = folders.AsyncGet 750 
                        match dir with 
                        | Some s ->  
                          let! fs = Directory.AsyncGetFiles(s, "*.*", SearchOption.AllDirectories) 
                          do! [ for z in fs -> printfn "%A" z; files.AsyncAdd(Some z) ]  
                                |> Async.Parallel |> Async.Ignore  
                          return! loop() 
                        | None -> printfn "Done!"; () } 
              do! loop() 
              printfn "after" // Never gets this far...  
              do! files.AsyncAdd(None) } 

    let rec printFiles() = 
      async { let! file = files.AsyncGet()
              match file with 
              | Some s ->  
                printfn "%s" s 
                return! printFiles() 
              | None -> () } 

    //let crawler = 
    //  MailboxProcessor.Start(fun self -> 
    //    let rec loop() = 
    //      async { let! dir = self.Receive() 
    //              folderCollector.Post dir 
    //              do! Async.StartChild( 
    //                    async { let! dirs = Directory.AsyncGetDirectories dir 
    //                            for z in dirs do self.Post z }) |> Async.Ignore 
    //              return! loop() } 
    //    loop()) 
    // 
    //crawler.Post @"C:\Projects" 
 
    let cts = new CancellationTokenSource() 
    Async.Start(folderCollector @"C:\Projects", cts.Token) 
    Async.Start(fileCollector(), cts.Token) 
    Async.RunSynchronously(printFiles(), cancellationToken = cts.Token) 
 
    printfn "DONE"
    ////


    let rec quicksort = function
        | [] -> []
        | pivot::rest ->
            let hi,lo = rest |> List.partition(fun x -> x < pivot)
            quicksort lo @ [rest] @ quicksort hi
