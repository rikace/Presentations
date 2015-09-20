namespace Easj360FSharpLib 

open System
open System.IO
open System.Collections.Generic
open System.Threading
open Microsoft.VisualBasic.FileIO

module Agents =
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


    type internal ThrottlingAgentMessage = 
      | Completed
      | Work of Async<unit>
    
    type ThrottlingAgent(limit) = 
      let agent = MailboxProcessor.Start(fun agent -> 

        let rec waiting () = 
          agent.Scan(function
            | Completed -> Some(working (limit - 1))
            | _ -> None)

        and working count = async { 
          let! msg = agent.Receive()
          match msg with 
          | Completed -> 
              return! working (count - 1)
          | Work work ->
              async { try do! work 
                      finally agent.Post(Completed) }
              |> Async.Start
              if count < limit then return! working (count + 1)
              else return! waiting () }
        working 0)      

      member x.DoWork(work) = agent.Post(Work work)

module IOperations =
    type IOperations(source:string, destination) =
        member x.CreatedOrChangedFunc (m:FileSystemEventArgs) = 
           if Directory.Exists(m.FullPath) then FileSystem.CopyDirectory(m.FullPath, m.FullPath.Replace(source, destination), true)
           elif File.Exists(m.FullPath) then FileSystem.CopyFile(m.FullPath, m.FullPath.Replace(source, destination), true) 

        member x.DeleteFunc (m:FileSystemEventArgs) =  
            let mp = m.FullPath.Replace(source, destination)
            if Directory.Exists(mp) then FileSystem.DeleteDirectory(mp, Microsoft.VisualBasic.FileIO.DeleteDirectoryOption.DeleteAllContents) 
            elif File.Exists(mp) then FileSystem.DeleteFile(mp)  

        member x.RenameFunc (m:#RenamedEventArgs) =              
            let oldName = m.OldFullPath.Replace(source, destination)
            let newName = m.FullPath.Replace(source, destination)
            if Directory.Exists(oldName) then FileSystem.RenameDirectory(oldName, m.Name)
            elif File.Exists(oldName) then FileSystem.RenameFile(oldName, Path.GetFileName(newName))
            elif Directory.Exists(m.FullPath) && (not(Directory.Exists(newName)) || not(Directory.Exists(oldName))) then FileSystem.CreateDirectory(newName)     

module FolderReplicatorV2 =
    open Agents
    open IOperations

    type EventSystemInfo = 
        { Path : string
          EventTime : DateTime
          EventIO:FileSystemEventArgs
          ChangeType : System.IO.WatcherChangeTypes }

    type RetryBuilder(max, delay:int) = 
      member x.Return(a) = a          
      member x.Delay(f) = f           
      member x.Zero() = failwith "Zero" 
      member x.Run(f) = let rec loop(n) = 
                            if n > 0 then
                                try f() 
                                with _ ->   System.Threading.Thread.Sleep(delay)
                                            loop(n-1)
                        loop max

    [<System.Security.Permissions.PermissionSet(System.Security.Permissions.SecurityAction.Demand, Name = "FullTrust")>]
    type public ReplicatorAgent(source : string, destination : string, filter : string, ?throttlinglimit : int, ?bulkSize : int, ?batchTimeout : int) = 
        let token = new System.Threading.CancellationTokenSource()
        let bulkSize = defaultArg bulkSize 10
        let batchTimeout = defaultArg batchTimeout 10000
        let ioOps = IOperations(source, destination)
        let queueAgent = new BlockingQueueAsyncAgent<EventSystemInfo list>()
        let throttlinglimit = defaultArg throttlinglimit 2
        let throttlingAgent = new ThrottlingAgent(throttlinglimit)
        let retry = RetryBuilder(10, 200)    
        let interval = new TimeSpan(0, 0, 0, 0, 150)

        let agent : Agent<_> = 
            Agent.Start((fun agent -> 
                        let rec loop remainingTime messages  = 
                            async { let start = DateTime.Now
                                    let! msg = agent.TryReceive(timeout = max 0 remainingTime)
                                    let elapsed =  int (DateTime.Now - start).TotalMilliseconds
                                    match msg with
                                    | Some(m) ->    let messageFiltered = (m :: messages) 
                                                    if List.length messageFiltered = bulkSize - 1 then 
                                                        do! queueAgent.AsyncAdd(messageFiltered)
                                                        return! loop batchTimeout [] 
                                                    else 
                                                        return! loop (remainingTime - elapsed) (messageFiltered) 
                                    | None when List.length messages <> 0 ->    do! queueAgent.AsyncAdd(messages)                                                                            
                                                                                return! loop batchTimeout [] 
                                    | None ->  return! loop batchTimeout [] }
                        loop batchTimeout [] ), cancellationToken = token.Token)
    
        let startToConsumeBatch = 
            async { while true do
                        let! batch = queueAgent.AsyncGet()
                        throttlingAgent.DoWork(async {
                            batch |> List.sortBy(fun f -> f.EventTime)
                                  |> List.iter(fun f -> retry { match f with                                   
                                                                | {ChangeType=System.IO.WatcherChangeTypes.Deleted} -> ioOps.DeleteFunc f.EventIO
                                                                | {ChangeType=System.IO.WatcherChangeTypes.Renamed} -> ioOps.RenameFunc (f.EventIO :?> RenamedEventArgs)
                                                                | _ -> ioOps.CreatedOrChangedFunc f.EventIO }) }) }

        let systemWatcher() = 
            let fsw = new FileSystemWatcher()
            fsw.BeginInit()
            fsw.InternalBufferSize <- 1024 * 64
            fsw.Path <- source
            fsw.Filter <- filter
            fsw.NotifyFilter <- NotifyFilters.LastWrite ||| NotifyFilters.DirectoryName ||| NotifyFilters.FileName
            fsw.IncludeSubdirectories <- true
            fsw.EndInit()       
            (fsw, { new System.IDisposable with
                       member x.Dispose() = 
                           fsw.EnableRaisingEvents <- false
                           fsw.Dispose() })
    
        let fswChangeEvent (ev : IObservable<#FileSystemEventArgs>) = 
            ev  |> Observable.map (fun f -> 
                       { Path = f.FullPath
                         EventIO = f
                         EventTime = DateTime.UtcNow
                         ChangeType = f.ChangeType})
                |> Observable.subscribe (agent.Post)
    
    //    let fswRenameddEvent (ev : IObservable<RenamedEventHandler, RenamedEventArgs>) = 
    //        ev  |> Observable.map (fun f -> 
    //                   { Path = f.FullPath
    //                     EventIO = f
    //                     EventTime = DateTime.UtcNow
    //                     ChangeType = f.ChangeType })
    //            |> Observable.subscribe (agent.Post)
    
        let fsws = [|  systemWatcher() |> (fun (w, d) -> (w.Changed |> fswChangeEvent), w)
                       systemWatcher() |> (fun (w, d) -> (w.Created |> fswChangeEvent), w)
                       systemWatcher() |> (fun (w, d) -> (w.Deleted |> fswChangeEvent), w)
                       systemWatcher() |> (fun (w, d) -> ((w.Renamed :?> IObservable<FileSystemEventArgs>) |> fswChangeEvent), w) |]

        new(source:string, destination:string) =
                ReplicatorAgent(source, destination, "*.*", 1, 20, 5000)

        member public x.Start() =
                if not(Directory.Exists(source)) then failwith "Source Directory not Exists"
                if not(Directory.Exists(destination)) then Directory.CreateDirectory(destination) |> ignore
                fsws |> Array.iter (fun (i, f) -> f.EnableRaisingEvents <- true)
                Async.Start(startToConsumeBatch, token.Token)
            
        member public x.Stop() =                                       
                fsws |> Array.iter (fun (i, d)-> i.Dispose(); d.Dispose())
                token.Cancel()                              