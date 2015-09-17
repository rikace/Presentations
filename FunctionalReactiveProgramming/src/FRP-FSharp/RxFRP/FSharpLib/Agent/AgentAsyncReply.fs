namespace Easj360FSharp 

module AgentAsyncReply =

    type countMsg =
    | Die
    | Incr of int
    | GetCounter of AsyncReplyChannel<int>
    | Fetch of int * AsyncReplyChannel<int>

    type counter() =
        let innerCounter =
            MailboxProcessor.Start(fun inbox ->
                let rec loop n =
                    async { let! msg = inbox.Receive()
                            match msg with
                            | Die -> return ()
                            | GetCounter(reply) ->  reply.Reply(n)
                                                    return! loop(n + 1)
                            | Incr x -> return! loop(n + x)
                            | Fetch(x, reply) ->
                                let res = x * x
                                reply.Reply(res)
                                return! loop(n + x)}
                loop 0)
    
        member this.Incr(x) = innerCounter.Post(Incr x)
        member this.Fetch(x) = innerCounter.PostAndAsyncReply((fun reply -> Fetch(x, reply)), timeout = 2000)
        member this.Die() = innerCounter.Post(Die)
        member this.GetCounter() = innerCounter.PostAndAsyncReply((fun reply -> GetCounter(reply)), timeout = 2000)
        
        interface System.IDisposable with
            member this.Dispose() = this.Die()


