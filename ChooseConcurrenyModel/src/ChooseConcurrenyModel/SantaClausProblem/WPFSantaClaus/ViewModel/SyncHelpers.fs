namespace SantaClausProblem

open System
open System.Threading

[<AutoOpenAttribute>]
module SyncHelpersModule =
        type Agent<'T> = MailboxProcessor<'T>

        type SynGateMessage =
            | AquireLock of AsyncReplyChannel<unit>
            | Release
           
        // SynGate uses a MailboxProcessor to emulate the synchronization
        // lock of a Semaphore maintaing the asynchronous semantic of F# Async Workflow
        type SyncGate(locks, cancellationToken, ?continuation:unit -> unit) =
            let continuation = defaultArg continuation (fun () -> ())
            let agent = Agent.Start((fun inbox ->
                    let rec aquiringLock n  = async {  
                        let! msg = inbox.Receive()
                        match msg with
                        | AquireLock(reply) ->  reply.Reply()
                                         // check if the number of locks aquired 
                                         // has reached the expected number, if so,  
                                         // the internal state changes to wait the relase of a lock 
                                         // before continuing 
                                                if n < locks - 1 then return! aquiringLock (n + 1)
                                                else return! releasingLock()
                        | Release ->    return! aquiringLock (n - 1) }
                     and releasingLock() =  
                         inbox.Scan(function
                                      | Release -> Some(aquiringLock(locks - 1))
                                      | _ -> None)
                    aquiringLock 0),cancellationToken)
            interface IDisposable with
                member x.Dispose() = (agent :> IDisposable).Dispose()
            member x.AquireAsync() = agent.PostAndAsyncReply AquireLock
            member x.Release() = agent.Post Release

        // The BarrierAsync is a block synchronization mechanism, which uses a MailboxProcessor with 
        // asynchronous message passing semantic to process messages sequentially instead of using 
        // low-level concurrency lock primitives.
        // The BarrierAsync blocks the threads until it reaches the number of
        // signals excpeted, then it replies to all the workers releasing them to continue the work
        type BarrierAsync(workers, cancellationToken, ?continuation:unit -> unit) =
             let continuation = defaultArg continuation (fun () -> ())
             let agent = Agent.Start((fun inbox ->
                            let rec loop replies = async {
                                    let! (reply:AsyncReplyChannel<int>) = inbox.Receive()
                                    let replies = reply::replies
                            // check if the number of workers waiting for a reply to continue
                            // has reached the expected number, if so, the reply functions 
                            // must be invoked for all the workers waiting to be resumed  
                            // before continuing and restaring with empty reply workers
                                    if (replies.Length) = workers then
                                        replies |> List.iteri(fun index reply -> reply.Reply(index))
                                        continuation()
                                        return! loop []
                                    else return! loop replies }
                            loop []),cancellationToken)
             interface IDisposable with
                member x.Dispose() = (agent :> IDisposable).Dispose()
             member x.AsyncSignalAndWait() = agent.PostAndAsyncReply(fun reply -> reply)  



