#r "FSharp.PowerPack.dll"
#load "..\CommonModule.fsx"
#load "..\Utilities\show.fs"

open System
open System.IO
open System.Threading
open Microsoft.FSharp.Control
open System.Collections.Generic
open Common

type internal SyncGateStates =
    | Free
    | OwnedByReaders
    | OwnedByReadersAndWriterPending
    | OwnedByWriter
    | ReservedForWriter
 
type SyncGateMode =
    | Exclusive
    | Shared
 
type SyncGate(blockReadersUntilFirstWriteCompletes : bool) =
    let lockObj = new Object()
    let writeRequests = new Queue<IDisposable -> unit>()
    let readRequests = new Queue<IDisposable -> unit>()
    let conditional condition trueVal falseVal =
        if condition then
            trueVal
        else
            falseVal
    let mutable state = conditional blockReadersUntilFirstWriteCompletes ReservedForWriter Free
    let mutable numReaders = 0
 
    member this.EndRegion (mode : SyncGateMode) =
        lock lockObj (fun () ->
            let mutable processQueues = false
            match mode with
            | Shared ->
                numReaders <- numReaders - 1
                processQueues <- (numReaders <= 0)
            | Exclusive -> processQueues <- true
 
            if processQueues then
                if writeRequests.Count > 0 then
                    state <- OwnedByWriter
                    this.QueueContinuation(writeRequests.Dequeue(), Exclusive)
                elif readRequests.Count > 0 then
                    state <- OwnedByReaders
                    numReaders <- readRequests.Count
                    while readRequests.Count > 0 do
                        this.QueueContinuation(readRequests.Dequeue(), Shared)
                else
                    state <- Free
        )

    member private this.QueueContinuation(cont, mode) =
        ThreadPool.QueueUserWorkItem (fun (_) -> cont(
                                                        { new System.IDisposable with
                                                            member d.Dispose() =
                                                                this.EndRegion mode }
                                                        )
                                     ) |> ignore
  
    member this.BeginRegion mode =
        Async.FromContinuations(fun (cont,_,_) ->
            lock lockObj (fun () ->
                match mode with
                | Exclusive ->
                    match state with
                    | Free | ReservedForWriter ->
                        state <- OwnedByWriter
                        this.QueueContinuation(cont, mode)
                    | OwnedByReaders | OwnedByReadersAndWriterPending ->
                        state <- OwnedByReadersAndWriterPending
                        writeRequests.Enqueue(cont)
                    | OwnedByWriter ->
                        writeRequests.Enqueue(cont)
                | Shared ->
                    match state with
                    | Free | OwnedByReaders ->
                        state <- OwnedByReaders
                        numReaders <- numReaders + 1
                        this.QueueContinuation(cont, mode)
                    | OwnedByReadersAndWriterPending | OwnedByWriter | ReservedForWriter ->
                        readRequests.Enqueue(cont)   
            )
        )


(********************************************************************************************************)
(******************************* T E S T ****************************************************************)
(********************************************************************************************************)

module TestSyncGate =
    let filePathSource = Path.Combine(Path.GetDirectoryName(__SOURCE_DIRECTORY__), "Data", "test.txt")   

    let asyncGate = new SyncGate(true)

    let sb = new System.Text.StringBuilder()

    let readFile filePath = async {
        use! x = asyncGate.BeginRegion SyncGateMode.Shared
        use stream = new FileStream(filePath, FileMode.Append, FileAccess.Write, FileShare.Write, 0x1000, FileOptions.Asynchronous)
        use reader = new StreamReader(stream)
        return! reader.AsyncReadToEnd() }

    let writeInMemory (text:string)= async {
        use! x = asyncGate.BeginRegion SyncGateMode.Exclusive
        sb.Append(text) |> ignore }
    
    let process' = async {
        let! text = readFile filePathSource
        do! writeInMemory  text }

    [ for _ in [0..20] -> process'] |> Async.Parallel |> Async.RunSynchronously
    
    sb.ToString() |> show


    
 