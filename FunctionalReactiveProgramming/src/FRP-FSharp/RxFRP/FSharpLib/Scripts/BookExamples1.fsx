#load "AgentSystem.fs"
open AgentSystem.LAgent

type msg1 = Message1 | Message2 of int | Message3 of string
            
let a = MailboxProcessor.SpawnParallelWorker(function
                | Message1 -> printfn "Message1";
                | Message2 n -> printfn "Message2 %i" n;
                | Message3 _ -> failwith "I failed"
                , 10
                , errorHandler = (fun ex _ -> printfn "%A" ex; ContinueProcessing()))


a.Post(Message1)
a.Post(Message2(100))
a.Post(Message3("abc"))
a.Post(Message2(100))


let echo = MailboxProcessor<_>.SpawnWorker(fun msg -> printfn "%s" msg)
echo.Post("Hello")

let counter1 = MailboxProcessor.SpawnAgent((fun i (n:int) -> printfn "n = %d, waiting..." n; n + i), 0)
counter1.Post(10)
counter1.Post(30)
counter1.Post(20)

type msg = Increment of int | Fetch of AsyncReplyChannel<int> | Stop

exception StopException

type CountingAgent() =
    let counter = MailboxProcessor.SpawnAgent((fun msg n ->
                    match msg with
                    | Increment m ->  n + m
                    | Stop -> raise(StopException)
                    | Fetch replyChannel ->
                        do replyChannel.Reply(n)
                        n
                  ), 0, errorHandler = (fun _ _ _ -> StopProcessing))
    member a.Increment(n) = counter.Post(Increment(n))
    member a.Stop() = counter.Post(Stop)
    member a.Fetch() = counter.PostAndReply(fun replyChannel -> Fetch(replyChannel))    
        
let counter2 = CountingAgent()
counter2.Increment(1)
counter2.Fetch()
counter2.Increment(2)
counter2.Fetch()
counter2.Stop()                             