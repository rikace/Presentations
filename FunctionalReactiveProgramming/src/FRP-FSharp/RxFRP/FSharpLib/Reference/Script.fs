namespace ScriptsVary

//#light
open System
open System.Linq
open System.Windows.Forms
open System.Threading
open System.IO
open System.Threading.Tasks

module srcModule =

    let megabyte  = 1024I    * 1024I
    let gigabyte  = megabyte * 1024I
    let terabyte  = gigabyte * 1024I
    let petabyte  = terabyte * 1024I
    let exabyte   = petabyte * 1024I
    let zettabyte = exabyte  * 1024I

    type LazyBinTree<'a> =
        | Node of 'a * LazyBinTree<'a> Lazy * LazyBinTree<'a> Lazy
        | Empty
    
    let rec map f tree =
        match tree with
        | Empty -> Empty
        | Node(x, l, r) ->
            Node(
                f x, 
                lazy(
                    let lfNode = l.Value
                    map f lfNode
                ), 
                lazy(
                    let rtNode = r.Value
                    map f rtNode
                )
            )


    type BinTree<'a> =
        | Node of 'a * BinTree<'a> * BinTree<'a>
        | Empty

    let rec iterNonTR f binTree =
        match binTree with
        | Empty -> ()
        | Node(x, l, r) ->
            f x       
            iterNonTR f l  // NOT in tail position
            iterNonTR f r  // In tail position

    let square x = x * x

    let rec sumSquares nums =
        match nums with
        | []    -> 0
        | h::t  -> square h + sumSquares t


    let pmap f (l:seq<_>) =
        let pe = l.AsParallel<_>()
        ParallelEnumerable.Select(pe, Func<_,_>(f))

    let psum (l:seq<_>) =
        let pe = l.AsParallel<float>()
        ParallelEnumerable.Sum(pe)

    let grid (prices:seq<System.DateTime * float>) =
        let form = new Form(Visible = true, TopMost = true)
        let grid = new DataGridView(Dock = DockStyle.Fill, Visible = true)

        form.Controls.Add(grid)
        grid.DataSource <- prices |> Seq.toArray


    ///////////////////

    // Creating new threads
    // What will execute on each thread
    let threadBody() =
        for i in 1 .. 5 do
            // Wait 1/10 of a second
            Thread.Sleep(100)
            printfn "[Thread %d] %d..." 
                Thread.CurrentThread.ManagedThreadId
                i

    let spawnThread() =
        let thread = new Thread(threadBody)
        thread.Start()
    
    // Spawn a couple of threads at once
    spawnThread()
    spawnThread()

    // ----------------------------------------------------------------------------

    ThreadPool.QueueUserWorkItem(fun _ -> for i = 1 to 5 do printfn "%d" i)

    // Our thread pool task, note that the delegate's
    // parameter is of type obj
    let printNumbers (max : obj) =
        for i = 1 to (max :?> int) do
            printfn "%d" i

    ThreadPool.QueueUserWorkItem(new WaitCallback(printNumbers), box 5)

    // ----------------------------------------------------------------------------

    let sumArray (arr : int[]) =
        let total = ref 0

        // Add the first half
        let thread1Finished = ref false

        ThreadPool.QueueUserWorkItem(
            fun _ -> for i = 0 to arr.Length / 2 - 1 do
                        total := arr.[i] + !total
                     thread1Finished := true
            ) |> ignore

        // Add the second half
        let thread2Finished = ref false

        ThreadPool.QueueUserWorkItem(
            fun _ -> for i = arr.Length / 2 to arr.Length - 1 do
                        total := arr.[i] + !total
                     thread2Finished := true
            ) |> ignore

        // Wait while the two threads finish their work
        while !thread1Finished = false ||
              !thread2Finished = false do

              Thread.Sleep(0)

        !total

    // ----------------------------------------------------------------------------

    let lockedSumArray (arr : int[]) =
        let total = ref 0

        // Add the first half
        let thread1Finished = ref false
        ThreadPool.QueueUserWorkItem(
            fun _ -> for i = 0 to arr.Length / 2 - 1 do
                        lock (total) (fun () -> total := arr.[i] + !total)
                     thread1Finished := true
            ) |> ignore

        // Add the second half
        let thread2Finished = ref false
        ThreadPool.QueueUserWorkItem(
            fun _ -> for i = arr.Length / 2 to arr.Length - 1 do
                        lock (total) (fun () -> total := arr.[i] + !total)
                     thread2Finished := true
            ) |> ignore

        // Wait while the two threads finish their work
        while !thread1Finished = false ||
              !thread2Finished = false do

              Thread.Sleep(0)

        !total

    // ----------------------------------------------------------------------------

    type BankAccount = { AccountID : int; OwnerName : string; mutable Balance : int }

    /// Transfer money between bank accounts
    let transferFunds amount fromAcct toAcct =

        printfn "Locking %s's account to deposit funds..." toAcct.OwnerName
        lock fromAcct
            (fun () ->
                printfn "Locking %s's account to withdrawl funds..." fromAcct.OwnerName
                lock toAcct
                    (fun () -> 
                        fromAcct.Balance <- fromAcct.Balance - amount
                        toAcct.Balance   <- toAcct.Balance + amount
                    )
            )

    // ----------------------------------------------------------------------------
        
    let processFileAsync (filePath : string) (processBytes : byte[] -> byte[]) =

        // This is the callback from when the AsyncWrite completes
        let asyncWriteCallback = 
            new AsyncCallback(fun (iar : IAsyncResult) ->
                 // Get state from the async result
                let writeStream = iar.AsyncState :?> FileStream
            
                // End the async write operation by calling EndWrite
                let bytesWritten = writeStream.EndWrite(iar)
                writeStream.Close()
            
                printfn 
                    "Finished processing file [%s]" 
                    (Path.GetFileName(writeStream.Name))
            )
    
        // This is the callback from when the AsyncRead completes
        let asyncReadCallback = 
            new AsyncCallback(fun (iar : IAsyncResult) -> 
                // Get state from the async result
                let readStream, data = iar.AsyncState :?> (FileStream * byte[])
            
                // End the async read by calling EndRead
                let bytesRead = readStream.EndRead(iar)
                readStream.Close()
            
                // Process the result
                printfn 
                    "Processing file [%s], read [%d] bytes" 
                    (Path.GetFileName(readStream.Name))
                    bytesRead
                
                let updatedBytes = processBytes data
            
                let resultFile = new FileStream(readStream.Name + ".result",
                                               FileMode.Create)
            
                let _ = 
                    resultFile.BeginWrite(
                        updatedBytes, 
                        0, updatedBytes.Length, 
                        asyncWriteCallback, 
                        resultFile)
                    
                ()
            )

        // Begin the async read, whose callback will begin the async write
        let fileStream = new FileStream(filePath, FileMode.Open, FileAccess.Read, 
                                        FileShare.Read, 2048,
                                        FileOptions.Asynchronous)

        let fileLength = int fileStream.Length
        let buffer = Array.zeroCreate fileLength

        // State passed into the async read
        let state = (fileStream, buffer)
    
        printfn "Processing file [%s]" (Path.GetFileName(filePath))
        let _ = fileStream.BeginRead(buffer, 0, buffer.Length, 
                                     asyncReadCallback, state)
        ()

    // ----------------------------------------------------------------------------

    open System.IO

    let asyncProcessFile (filePath : string) (processBytes : byte[] -> byte[]) =
        async {
        
            printfn "Processing file [%s]" (Path.GetFileName(filePath))
        
            use fileStream = new FileStream(filePath, FileMode.Open)
            let bytesToRead = int fileStream.Length
        
            let! data = fileStream.AsyncRead(bytesToRead)
        
            printfn 
                "Opened [%s], read [%d] bytes" 
                (Path.GetFileName(filePath)) 
                data.Length
        
            let data' = processBytes data
        
            use resultFile = new FileStream(filePath + ".results", FileMode.Create)
            do! resultFile.AsyncWrite(data', 0, data'.Length)
        
            printfn "Finished processing file [%s]" <| Path.GetFileName(filePath)
        } |> Async.Start

    // ----------------------------------------------------------------------------

    //#r "FSharp.PowerPack.dll"

    open System.IO
    open System.Net
    open Microsoft.FSharp.Control.WebExtensions

    let getHtml (url : string) =
        async {

            let req = WebRequest.Create(url)
            let! rsp = req.AsyncGetResponse()

            use stream = rsp.GetResponseStream()
            use reader = new StreamReader(stream)

            //return! reader.AsyncReadToEnd()
            return reader.ReadToEnd()
        }
    
    let html =
        getHtml "http://en.wikipedia.org/wiki/F_Sharp_programming_language"
        |> Async.RunSynchronously

    // ----------------------------------------------------------------------------

    let asyncTaskX = async { failwith "error" }

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    asyncTaskX
        |> Async.Catch 
        |> Async.RunSynchronously
        |> function 
           | Choice1Of2 result     -> printfn "Async operation completed: %A" result
           | Choice2Of2 (ex : exn) -> printfn "Exception thrown: %s" ex.Message

    // ----------------------------------------------------------------------------

    open System
    open System.Threading

    let cancelableTask =
        async {
            printfn "Waiting 10 seconds..."
            for i = 1 to 10 do 
                printfn "%d..." i
                do! Async.Sleep(1000)
            printfn "Finished!"
        }
   
    // Callback used when the operation is canceled
    let cancelHandler (ex : OperationCanceledException) = 
        printfn "The task has been canceled."

    Async.TryCancelled(cancelableTask, cancelHandler)
     |> Async.Start

    // ...

    Async.CancelDefaultToken()

    // ----------------------------------------------------------------------------

    let superAwesomeAsyncTask = async { return 5 }

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Async.StartWithContinuations(
        superAwesomeAsyncTask,
        (fun (result : int) -> printfn "Task was completed with result %d" result),
        (fun (exn : Exception) -> printfn "threw exception"),
        (fun (oce : OperationCanceledException) -> printfn "OCE")
    )

    // ----------------------------------------------------------------------------

    open System.Threading

    let computation = Async.TryCancelled(cancelableTask, cancelHandler)
    let cancellationSource = new CancellationTokenSource()

    Async.Start(computation, cancellationSource.Token)

    // ...

    cancellationSource.Cancel()

    // ----------------------------------------------------------------------------

//    open System.IO
//    open System.Net

    open Microsoft.FSharp.Control.WebExtensions

    let getHtml' (url : string) =
        async {

            let req = WebRequest.Create(url)
            let! rsp = req.AsyncGetResponse()

            use stream = rsp.GetResponseStream()
            use reader = new StreamReader(stream)

            return reader.ReadToEnd()
        }

    // ----------------------------------------------------------------------------

//    #r "System.Windows.Forms.dll"
//    #r "System.Drawing.dll"

//    open System.Threading
//    open System.Windows.Forms
//
//    let form = new Form(TopMost = true)
//
//    let pb   = new ProgressBar(Minimum = 0, Maximum = 15, Dock = DockStyle.Fill)
//    form.Controls.Add(pb)
//
//    form.Show()
//
//    async {
//        for i = 0 to 15 do
//            do! Async.Sleep(1000)
//            pb.Value <- i
//    } |> Async.Start

    // ----------------------------------------------------------------------------

   // #r "System.Core.dll" // Shouldn't be required post Beta2

//    open System
//    open System.IO

    type System.IO.Directory with
        /// Retrieve all files under a path asynchronously
        static member AsyncGetFiles(path : string, searchPattern : string) =
            let dele = new Func<string * string, string[]>(Directory.GetFiles)
            Async.FromBeginEnd(
                (path, searchPattern), 
                dele.BeginInvoke, 
                dele.EndInvoke)
        
    type System.IO.File with
        /// Copy a file asynchronously
        static member AsyncCopy(source : string, dest : string) =
            let dele = new Func<string * string, unit>(File.Copy)
            Async.FromBeginEnd((source, dest), dele.BeginInvoke, dele.EndInvoke)


    let asyncBackup path searchPattern destPath =
        async {
            let! files = Directory.AsyncGetFiles(path, searchPattern)
        
            for file in files do
                let filename = Path.GetFileName(file)
                do! File.AsyncCopy(file, Path.Combine(destPath, filename))
        }


//////////////////////////////////////////


    let agentBatch f = MailboxProcessor<int>.Start(fun i ->
                    let rec loop (c, lst) = async {
                        let! msg = i.Receive()
                        let newLst = msg::lst
                        if List.length newLst = 100 then
                            f(newLst)
                            return! loop (0, [])
                        return! loop ((c + 1), newLst)
                    }
                    loop (0, []))
                
    let agent = agentBatch (fun newLst -> ignore(Async.RunSynchronously( Async.StartChild(async { newLst |> List.rev |> List.iter (fun i -> printfn "%d" i) }))))

    for i in [1..1000] do agent.Post(i)                

    open System

    type BatchProcessingAgent<'T> (batchSize, timeout) = 
        let batchEvent = new Event<'T[]>()
        let agent : MailboxProcessor<'T> = MailboxProcessor.Start(fun agent -> 
        
            let rec loop remainingTime messages = async {
                let start = DateTime.Now
                let! msg = agent.TryReceive(timeout = max 0 remainingTime)
                let elapsed = int (DateTime.Now - start).TotalMilliseconds
                match msg with 
                | Some(msg) when List.length messages = batchSize - 1 -> batchEvent.Trigger(msg :: messages |> List.rev |> Array.ofList)
                                                                         return! loop timeout []
                | Some(msg) -> return! loop (remainingTime - elapsed) (msg::messages)
                | None when List.length messages <> 0 ->  batchEvent.Trigger(messages |> List.rev |> Array.ofList)
                                                          return! loop timeout []
                | None ->  return! loop timeout [] }
            loop timeout [] )

        member x.BatchProduced = batchEvent.Publish
        member x.Enqueue(v) = agent.Post(v)


    open System 
    open System.Drawing
    open System.Windows.Forms

    let frm = new Form()
    let lbl = new Label(Dock = DockStyle.Fill)
    frm.Controls.Add(lbl)
    frm.Show()

    // Create agent for bulking KeyPress events
    let ag = new BatchProcessingAgent<_>(5, 5000)
    frm.KeyPress.Add(fun e -> ag.Enqueue(e.KeyChar))
    ag.BatchProduced
        |> Event.map (fun chars -> new String(chars))
        |> Event.scan (+) ""
        |> Event.add (fun str -> lbl.Text <- str)



    // ----------------------------------------------------------------------------
    // Blocking queue agent
    // ----------------------------------------------------------------------------

    open System
    open System.Collections.Generic

    type Agent<'T> = MailboxProcessor<'T>

    type internal BlockingAgentMessage<'T> = 
      | Add of 'T * AsyncReplyChannel<unit> 
      | Get of AsyncReplyChannel<'T>

    /// Agent that implements an asynchronous blocking queue
    type BlockingQueueAgent<'T>(maxLength) =
      let agent = Agent.Start(fun agent ->
    
        let queue = new Queue<_>()

        let rec emptyQueue() = 
          agent.Scan(fun msg ->
            match msg with 
            | Add(value, reply) -> Some(enqueueAndContinue(value, reply))
            | _ -> None )
        and fullQueue() = 
          agent.Scan(fun msg ->
            match msg with 
            | Get(reply) -> Some(dequeueAndContinue(reply))
            | _ -> None )
        and runningQueue() = async {
          let! msg = agent.Receive()
          match msg with 
          | Add(value, reply) -> return! enqueueAndContinue(value, reply)
          | Get(reply) -> return! dequeueAndContinue(reply) }

        and enqueueAndContinue (value, reply) = async {
          reply.Reply() 
          queue.Enqueue(value)
          return! chooseState() }
        and dequeueAndContinue (reply) = async {
          reply.Reply(queue.Dequeue())
          return! chooseState() }
        and chooseState() = 
          if queue.Count = 0 then emptyQueue()
          elif queue.Count < maxLength then runningQueue()
          else fullQueue()

        // Start with an empty queue
        emptyQueue() )

      /// Asynchronously adds item to the queue. The operation ends when
      /// there is a place for the item. If the queue is full, the operation
      /// will block until some items are removed.
      member x.AsyncAdd(v:'T, ?timeout) = 
        agent.PostAndAsyncReply((fun ch -> Add(v, ch)), ?timeout=timeout)

      /// Asynchronously gets item from the queue. If there are no items
      /// in the queue, the operation will block unitl items are added.
      member x.AsyncGet(?timeout) = 
        agent.PostAndAsyncReply(Get, ?timeout=timeout)


    // ----------------------------------------------------------------------------

    let ag2 = new BlockingQueueAgent<int>(3)

    async { 
      for i in 0 .. 10 do 
        do! ag2.AsyncAdd(i)
        printfn "Added %d" i } |> Async.Start

    async { 
      while true do
        do! Async.Sleep(1000)
        let! v = ag2.AsyncGet()
        printfn "Got %d" v } |> Async.Start


    /////////////////////////////////////////

    let matrixMultiplyAsync (a:float[,]) (b:float[,]) =
        let rowsA, colsA = Array2D.length1 a, Array2D.length2 a
        let rowsB, colsB = Array2D.length1 b, Array2D.length2 b
        let result = Array2D.create rowsA colsB 0.0
        [ for i in 0 .. rowsA - 1 ->
            async {
               for j in 0 .. colsB - 1 do
                 for k in 0 .. colsA - 1 do
                   result.[i,j] <- result.[i,j] + a.[i,k] * b.[k,j]
            } ]
        |> Async.Parallel
        |> Async.RunSynchronously
        |> ignore
        result;;
    //val matrixMultiplyAsync : float [,] -> float [,] -> float [,]

    //matrixMultiplyAsync a b;;

    module testMatrix = 
        #r "FSharp.PowerPack.Parallel.Seq.dll"
        open Microsoft.FSharp.Collections

        let matrixMultiplyTasks (a:float[,]) (b:float[,]) =
            let rowsA, colsA = Array2D.length1 a, Array2D.length2 a
            let rowsB, colsB = Array2D.length1 b, Array2D.length2 b
            let result = Array2D.create rowsA colsB 0.0
            Parallel.For(0, rowsA, (fun i->
                for j = 0 to colsB - 1 do
                   for k = 0 to colsA - 1 do
                      result.[i,j] <- result.[i,j] + a.[i,k] * b.[k,j]))  
            |> ignore
            result;;
    //val matrixMultiplyTasks : float [,] -> float [,] -> float [,]

    //matrixMultiplyTasks a b;;
    module testMatrix2 = 
        #r "FSharp.PowerPack.Parallel.Seq.dll"
        open Microsoft.FSharp.Collections

        let matrixMultiplyPSeq (a:float[,]) (b:float[,]) =
            let rowsA, colsA = Array2D.length1 a, Array2D.length2 a
            let rowsB, colsB = Array2D.length1 b, Array2D.length2 b
            let result = Array2D.create rowsA colsB 0.0
            [ 0 .. rowsA - 1 ] |> PSeq.iter (fun i ->
              for j = 0 to colsB - 1 do
                for k = 0 to colsA - 1 do
                  result.[i,j] <- result.[i,j] + a.[i,k] * b.[k,j] )
            result;;
    //val matrixMultiplyPSeq : float [,] -> float [,] -> float [,]

    //matrixMultiplyPSeq a b;;


    ////////////////
    module testMatrix3 = 

    #r "System.Xml.Linq"
    #r "System.Core"
    open System
    open System.Linq
    open System.Xml
    open System.Xml.Linq
    open System.Net

    type FeedLink = 
      { Title : string
        Link : string
        Description : string }

    let xn s = XName.Get(s)

    let (?) (el:XElement) name = el.Element(xn name).Value

    let stripHtml (html:string) = 
        let res = System.Text.RegularExpressions.Regex.Replace(html, "<.*?>", "")
        if res.Length > 200 then res.Substring(0, 200) + "..." else res

    let AsyncDownloadFeed(url) = async { 
        let wc = new WebClient()
        let! rss = wc.AsyncDownloadString(Uri(url))
        let feed = XDocument.Parse(rss) 
        let elements = 
            [ for el in feed.Descendants(xn "item") do 
                  yield el?title, el?link ]
        return (url, elements) }

    type NewsPage = 
      { Feeds : seq<string * seq<FeedLink>> }

    let feeds = 
        [ "http://feeds.guardian.co.uk/theguardian/world/rss"
          "http://www.nytimes.com/services/xml/rss/nyt/GlobalHome.xml" 
          "http://feeds.bbci.co.uk/news/world/rss.xml" ]

//    let AsyncDownloadAll() = async {
//        let! results = 
//            [ for url in feeds -> AsyncDownloadFeed(url) ]
//            |> Async.Parallel
//        return { Feeds = results } }


////////////////////////////////

//open System.Data 
//open System.Data.SqlClient
//open Microsoft.FSharp.Reflection
//
//module Internal =
//    let createCommand name (args:'T) connection = 
//        let cmd = new SqlCommand(name, connection)
//        cmd.CommandType <- CommandType.StoredProcedure
//        SqlCommandBuilder.DeriveParameters(cmd)
//        let parameters = 
//          [| for (p:SqlParameter) in cmd.Parameters do
//                 if p.Direction = ParameterDirection.Input then
//                     yield p |]
//        let arguments = 
//            if typeof<'T> = typeof<unit> then [| |]
//            elif FSharpType.IsTuple(typeof<'T>) then
//                FSharpValue.GetTupleFields(args)
//            else [| args |]
//        if arguments.Length <> parameters.Length then 
//            failwith "Incorrect number of arguments!"
//        for (par, arg) in Seq.zip parameters arguments do 
//            par.Value <- arg
//        cmd
//
//
//    type RowReader(reader:SqlDataReader) = 
//        member private x.Reader = reader
//        static member (?) (x:RowReader, name:string) : 'R = 
//            x.Reader.[name] :?> 'R
//
//    type DatabaseQuery(connectionString:string) = 
//        member private x.ConnectionString = connectionString
//        static member (?) (x:DatabaseQuery, name) = fun (args:'T) -> seq {
//            use cn = new SqlConnection(x.ConnectionString)
//            cn.Open()
//            let cmd = Internal.createCommand name args cn
//            let reader = cmd.ExecuteReader()
//            let row = RowReader(reader)
//            while reader.Read() do yield row }
//
//    type DatabaseNonQuery(connectionString:string) = 
//        member x.ConnectionString = connectionString
//        static member (?) (x:DatabaseNonQuery, name) = fun (args:'T) -> 
//            use cn = new SqlConnection(x.ConnectionString)
//            cn.Open()
//            let cmd = Internal.createCommand name args cn
//            cmd.ExecuteNonQuery() |> ignore
//
//    type DynamicDatabase(connectionString:string) =
//        member x.Query = DatabaseQuery(connectionString)
//        member x.NonQuery = DatabaseNonQuery(connectionString)


