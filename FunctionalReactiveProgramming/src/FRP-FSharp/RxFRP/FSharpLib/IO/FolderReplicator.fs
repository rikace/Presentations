module FolderReplicator


//#r "Microsoft.VisualBasic"
open System
open System.IO
open System.Collections.Generic
open Microsoft.VisualBasic.FileIO
open System.Threading

type Agent<'a> = MailboxProcessor<'a>

    type internal BlockingAgentMessage<'T> = 
      | Add of 'T * AsyncReplyChannel<unit> 
      | Get of AsyncReplyChannel<'T>
    
    type BlockingQueueAsyncAgent<'T>() =      
      let agent = Agent.Start(fun agent ->
        let queue = new Queue<_>()
        let rec emptyQueue() = 
          agent.Scan(fun msg ->
            match msg with 
            | Add(value, reply) -> Some <| async {
                queue.Enqueue(value)
                reply.Reply()
                return! nonEmptyQueue() }
            | _ -> None )
        and nonEmptyQueue() = async {
          let! msg = agent.Receive()
          match msg with 
          | Add(value, reply) -> 
                queue.Enqueue(value)
                reply.Reply()
                return! nonEmptyQueue()
          | Get(reply) -> 
              let item = queue.Dequeue()
              reply.Reply(item)
              if queue.Count = 0 then return! emptyQueue()
              else return! nonEmptyQueue() }
        emptyQueue() )
 
      member x.AsyncAdd(v:'T, ?timeout) = 
        agent.PostAndAsyncReply((fun ch -> Add(v, ch)), ?timeout=timeout)

      member x.AsyncGet(?timeout) = 
        agent.PostAndAsyncReply(Get, ?timeout=timeout)            

[<System.Security.Permissions.PermissionSet(System.Security.Permissions.SecurityAction.Demand, Name="FullTrust")>]
type public ReplicatorAgent(source:string, destination:string, filter:string, ?throttlinglimit:int, ?bulkSize:int, ?batchTimeout:int) =
        let token = new System.Threading.CancellationTokenSource()
        let bulkSize = defaultArg bulkSize 10
        let batchTimeout = defaultArg batchTimeout 10000

        let queueAgent = new BlockingQueueAsyncAgent<(FileSystemEventArgs * DateTime) list>()

        [<LiteralAttribute>]        
        let retryProcess = 3
 
        [<LiteralAttribute>]
        let asyncSleep = 200
 
        let interval = new TimeSpan(0, 0, 0, 0, 250)

        let hasAnotherFileEventOccuredRecently =
             let lastFileEvent = new System.Collections.Concurrent.ConcurrentDictionary<string, DateTime>() 
             (fun (filePath:string) (currentTime:DateTime) ->
                 if lastFileEvent.ContainsKey(filePath) then
                     let lastEventTime = lastFileEvent.[filePath]
                     let timeSinceLastEvent = currentTime.Subtract(lastEventTime)
                     lastFileEvent.[filePath] <- currentTime
                     (timeSinceLastEvent.TotalMilliseconds > interval.TotalMilliseconds)
                 else
                     ignore(lastFileEvent.AddOrUpdate(filePath, currentTime, (fun s d -> d)) )
                     true)
 
        let CreatedOrChangedFunc (m:FileSystemEventArgs) = 
           if Directory.Exists(m.FullPath) then FileSystem.CopyDirectory(m.FullPath, m.FullPath.Replace(source, destination), true)
           elif File.Exists(m.FullPath) then FileSystem.CopyFile(m.FullPath, m.FullPath.Replace(source, destination), true) 

        let DeleteFunc (m:FileSystemEventArgs) =  
            let mp = m.FullPath.Replace(source, destination)
            if Directory.Exists(mp) then FileSystem.DeleteDirectory(mp, Microsoft.VisualBasic.FileIO.DeleteDirectoryOption.DeleteAllContents) 
            elif File.Exists(mp) then FileSystem.DeleteFile(mp)  

        let RenameFunc (m:RenamedEventArgs) =              
            let oldName = m.OldFullPath.Replace(source, destination)
            let newName = m.FullPath.Replace(source, destination)
            if Directory.Exists(oldName) then FileSystem.RenameDirectory(oldName, m.Name)
            elif File.Exists(oldName) then FileSystem.RenameFile(oldName, Path.GetFileName(newName))
            elif Directory.Exists(m.FullPath) && (not(Directory.Exists(newName)) || not(Directory.Exists(oldName))) then FileSystem.CreateDirectory(newName)     

        let agent : Agent<_> = 
            Agent.Start((fun agent -> 
                let rec loop remainingTime messages = async {
                  let start = DateTime.Now                   
                  let! msg = agent.TryReceive(timeout = max 0 remainingTime)
                  let elapsed = int (DateTime.Now - start).TotalMilliseconds
                  match msg with                                            
                  | Some(f, d) ->   if List.length messages = bulkSize - 1 then
                                       do! queueAgent.AsyncAdd((f, d) :: messages)
                                       return! loop batchTimeout []
                                    else return! loop (remainingTime - elapsed) ((f, d) :: messages)
                  | None when List.length messages <> 0 ->
                      do! queueAgent.AsyncAdd(messages)
                      return! loop batchTimeout [] 
                  | None -> return! loop batchTimeout [] }
                loop batchTimeout [] ), cancellationToken=token.Token)

        let startToConsumeBatch = async {
            while true do
                let! batch = queueAgent.AsyncGet()
                batch 
                |> List.sortBy(fun (_, d) -> d)
                |> List.map(fun (ac,_) -> 
                                let rec loop(r:int) = async { 
                                    try 
                                        match ac.ChangeType with
                                        | WatcherChangeTypes.Changed
                                        | WatcherChangeTypes.Created -> CreatedOrChangedFunc ac
                                        | WatcherChangeTypes.Deleted -> DeleteFunc ac
                                        | WatcherChangeTypes.Renamed -> RenameFunc (ac :?> RenamedEventArgs)
                                    with
                                    | _ ->  if retryProcess >= r then
                                                do! Async.Sleep(asyncSleep)
                                            return! loop(r + 1) } 
                                loop(0))
                |> Async.Parallel 
                |> Async.RunSynchronously
                |> ignore }

        let actionDuplicator (ev:IObservable<#FileSystemEventArgs>) = 
            ev  |> Observable.map (fun f -> (f, DateTime.Now))
                |> Observable.filter (fun (f, d) -> hasAnotherFileEventOccuredRecently (f.FullPath) d)
                |> Observable.subscribe(fun f -> agent.Post(f))

        let systemWatcher (f:FileSystemWatcher -> IDisposable) =
            let fsw = new FileSystemWatcher()
            fsw.BeginInit()
            fsw.InternalBufferSize <- 1024 * 64
            fsw.Path <- source
            fsw.Filter <- filter
            fsw.NotifyFilter <- NotifyFilters.LastWrite ||| NotifyFilters.DirectoryName ||| NotifyFilters.FileName
            fsw.IncludeSubdirectories <- true   
            let subIDisposable = f(fsw)
            fsw.EndInit()
            (fsw, { new System.IDisposable with
                                member x.Dispose() =
                                      fsw.EnableRaisingEvents <- false
                                      subIDisposable.Dispose()
                                      fsw.Dispose() })

        let fsws       = [| systemWatcher(fun w -> w.Changed |> actionDuplicator)
                            systemWatcher(fun w -> w.Created |> actionDuplicator)
                            systemWatcher(fun w -> w.Deleted |> actionDuplicator)
                            systemWatcher(fun w -> (w.Renamed :?> IObservable<FileSystemEventArgs>) |> actionDuplicator)                                 |]

        new(source:string, destination:string) =
            ReplicatorAgent(source, destination, "*.*", 1, 10, 15000)

        member public x.Start() =
            if not(Directory.Exists(source)) then
               failwith "Source Directory not Exists"
            if not(Directory.Exists(destination)) then
                Directory.CreateDirectory(destination) |> ignore
            fsws |> Array.iter (fun (f, _) -> f.EnableRaisingEvents <- true)
            Async.Start(startToConsumeBatch, token.Token)
            
        member public x.Stop() =                                       
            fsws |> Array.iter (fun (_, d)-> d.Dispose())
            token.Cancel()
//
//
//
//let test = new ReplicatorAgent(@"e:\Target", @"E:\Mirror")
//test.Start()
//test.Stop()
