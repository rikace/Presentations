#load "AgentSystem.fs"
open AgentSystem.LAgent

type internal msg = Increment of int | Fetch of AsyncReplyChannel<int> | Stop

type CountingAgent() =
    let counter = MailboxProcessor.Start(fun inbox ->
             // The states of the message-processing state machine...
             let rec loop n =
                async { let! msg = inbox.Receive()
                        match msg with
                        | Increment m ->
                            // increment and continue...
                            return! loop(n+m)
                        | Stop ->
                            // exit
                            return ()
                        | Fetch  replyChannel  ->
                            // post response to reply channel and continue
                            do replyChannel.Reply n
                            return! loop n }

             // The initial state of the message-processing state machine...
             loop(0))

    member a.Increment(n) = counter.Post(Increment n)
    member a.Stop() = counter.Post Stop
    member a.Fetch() = counter.PostAndReply(fun replyChannel -> Fetch replyChannel)

let counter = new CountingAgent()
counter.Increment(1)
counter.Fetch()
counter.Increment(2)
counter.Fetch()
counter.Stop()


let counter0 =
    MailboxProcessor.Start(fun inbox ->
        let rec loop n =
            async { 
                    let! msg = inbox.Receive()
                    return! loop(n+msg) }
        loop 0)

counter0.Post(3)

let counter1 = MailboxProcessor.SpawnAgent( (fun msg n -> msg + n), 0)

counter1.Start() // .Post(3)

