#load "Utils.fs"
open System.IO
open System
open System.Threading
open System.Collections.Generic
open System.Net
open AgentModule

// ===========================================
// Agent 101
// ===========================================
 
let cancellationToken = new CancellationTokenSource()

type internal MessageCounter = 
    | Increment of int 
    | Fetch of AsyncReplyChannel<int>
    | Operatiom of (int -> int) * AsyncReplyChannel<int>
    | Stop 
    | Pause
    | Resume

type CountingAgent() =

    let counter = Agent.Start((fun inbox ->
            
            let rec blocked(n) =           
                inbox.Scan(fun msg ->
                match msg with
                | Resume -> Some(async {
                    return! processing(n) })
                | _ -> None)

            and processing(n) = async {
                    let! msg = inbox.Receive()
                    match msg with
                    | Increment m -> return! processing(n + m)
                    | Stop -> ()
                    | Pause -> return! blocked(n)
                    | Resume -> return! processing(n)
                    | Operatiom(op, reply) -> 
                            let asyncOp = async {   let result = op(n)
                                                    do! Async.Sleep 5000
                                                    reply.Reply(result) }
                            Async.Start(asyncOp)
                            return! processing(n)
                    | Fetch replyChannel  ->    do replyChannel.Reply n
                                                return! processing(n) }
            processing(0)), cancellationToken.Token)

    member a.Increment(n) = counter.Post(Increment n)       
    member a.Stop() =   cancellationToken.Cancel()
                        //counter.Post Stop
    member a.Pause() = counter.Post Pause
    member a.Resume() = counter.Post Resume

    member a.Fetch() = counter.PostAndReply(fun replyChannel -> Fetch replyChannel)
    
    member a.FetchAsync(continuation) = 
            let opAsync = counter.PostAndAsyncReply(fun replyChannel -> Fetch replyChannel)

            Async.StartWithContinuations(opAsync, 
                (fun reply -> continuation reply), //continuation
                (fun _ -> ()), //exception
                (fun _ -> ())) //cancellation

    member a.Operation (f:(int -> int)) continuation =
            let opAsync = counter.PostAndAsyncReply(fun replyChannel -> Operatiom(f, replyChannel))
            
            Async.StartWithContinuations(opAsync, 
                (fun reply -> continuation reply), //continuation
                (fun _ -> ()), //exception
                (fun _ -> ())) //cancellation


let counterInc = new CountingAgent()


counterInc.Increment(1)
counterInc.Fetch()
counterInc.Pause()
counterInc.Increment(2)
counterInc.Resume()
counterInc.Fetch()

counterInc.FetchAsync(fun res -> printfn "Reply Async received: %d" res)

let add2 = (+) 2
counterInc.Operation(add2) (fun res -> printfn "Reply 'add2' received: %d" res)
counterInc.Fetch()

let mult3 = (*) 3
counterInc.Operation(add2) (fun res -> printfn "Reply 'mul3' received: %d" res)


counterInc.Fetch()
counterInc.Stop()
