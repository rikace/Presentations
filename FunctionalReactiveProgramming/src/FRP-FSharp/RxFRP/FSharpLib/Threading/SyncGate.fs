namespace Easj360FSharp

open System
open System.Threading
open System.Net
open System.IO
open System.Collections.Generic
open Microsoft.FSharp.Control


(*To use this implementation of SyncGate from an asynchronous workflow, you call BeginRegion passing Shared or Exclusive with the keyword use!. The thread from which BeginRegion is called is returned to the ThreadPool. When the lock becomes available, the continuation is queued to the ThreadPool for execution. That's where the rest of the workflow picks up. When the identifier you bound to the return value of BeginRegion goes out of scope, Dispose and thus EndRegion gets called automatically. If you need more control over when EndRegion gets called, you could bind with let! instead of use!.*)
type internal SyncGateStates =
    | Free
    | OwnedByReaders
    | OwnedByReadersAndWriterPending
    | OwnedByWriter
    | ReservedForWriter
 
type SyncGateMode = 
    | Exclusive
    | Shared
 
type FSharpSyncGate(blockReadersUntilFirstWriteCompletes : bool) =
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
 
    member private this.QueueContinuation(cont, mode) =
        ThreadPool.QueueUserWorkItem (fun (_) -> cont(  { new System.IDisposable with
                                                            member d.Dispose() = 
                                                                this.EndRegion mode } )) |> ignore

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
