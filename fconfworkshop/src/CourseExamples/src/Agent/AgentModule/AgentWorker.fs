module AgentWorker

open System
open System.Threading
open System.Collections.Generic
open AgentModule


type AfterError<'state> =
    | ContinueProcessing of 'state
    | StopProcessing
    | RestartProcessing

type MailboxProcessor<'a> with
    static member public SpawnAgent<'b>(messageHandler : 'a -> 'b -> 'b, 
                                        initialState : 'b, ?timeout : 'b -> int,
                                        ?timeoutHandler : 'b -> AfterError<'b>, 
                                        ?errorHandler : Exception -> 'a option -> 'b -> AfterError<'b>) 
                                        : MailboxProcessor<'a> =
        let timeout = defaultArg timeout (fun _ -> -1)
        let timeoutHandler = defaultArg timeoutHandler (fun state -> ContinueProcessing(state))
        let errorHandler = defaultArg errorHandler (fun _ _ state -> ContinueProcessing(state))
        
        MailboxProcessor.Start(fun inbox ->
            let rec loop (state) =
                async {
                    let! msg = inbox.TryReceive(timeout (state))
                    try
                        match msg with
                        | None ->
                            match timeoutHandler state with
                            | ContinueProcessing(newState) -> return! loop (newState)
                            | StopProcessing -> return ()
                            | RestartProcessing -> return! loop (initialState)
                        | Some(m) -> return! loop (messageHandler m state) 
                    with ex ->
                        match errorHandler ex msg state with
                        | ContinueProcessing(newState) -> return! loop (newState)
                        | StopProcessing -> return ()
                        | RestartProcessing -> return! loop (initialState)
                }
            loop (initialState))

type MailboxProcessor<'a> with    
    static member public SpawnWorker(messageHandler, ?timeout, ?timeoutHandler, ?errorHandler) =
        let timeout = defaultArg timeout (fun () -> -1)
        let timeoutHandler = defaultArg timeoutHandler (fun _ -> ContinueProcessing(()))
        let errorHandler = defaultArg errorHandler (fun _ _ -> ContinueProcessing(()))
        MailboxProcessor.SpawnAgent((fun msg _ ->
                                    messageHandler msg
                                    ()), (), timeout, timeoutHandler, (fun ex msg _ -> errorHandler ex msg))

type MailboxProcessor<'a> with
    static member public SpawnParallelWorker(messageHandler, 
                                             workerCount, 
                                             ?timeout, 
                                             ?timeoutHandler, 
                                             ?errorHandler) =
        let timeout = defaultArg timeout (fun () -> -1)
        let timeoutHandler = defaultArg timeoutHandler (fun _ -> ContinueProcessing(()))
        let errorHandler = defaultArg errorHandler (fun _ _ -> ContinueProcessing(()))
       
        MailboxProcessor<'a>
            .SpawnAgent((fun msg (workers : MailboxProcessor<'a> array, index) ->
                        workers.[index].Post msg
                        (workers, (index + 1) % workerCount)),
                        (Array.init workerCount
                                (fun _ -> MailboxProcessor<'a>.SpawnWorker(messageHandler, 
                                                                           timeout, 
                                                                           timeoutHandler, 
                                                                           errorHandler)), 0))






let agent = MailboxProcessor<_>.SpawnAgent((fun msg state ->
                                                match msg with
                                                | n when n % 2 = 0 -> let result = n + state
                                                                      printfn "sum result %d" result
                                                                      result
                                                | _ -> state / 0), 0, errorHandler=(fun ex msg s -> printfn "Error : %s" ex.Message
                                                                                                    ContinueProcessing(s)))

agent.Post 1
agent.Post 4
agent.Post 7
agent.Post 10



let echo = MailboxProcessor<_>.SpawnWorker(fun msg -> printfn "%s" msg)

echo.Post("Hello")
echo.Post("World")
echo.Post "Hello guys!"


type Message =
    | Message1
    | Message2 of int
    | Message3 of string

let a = MailboxProcessor.SpawnParallelWorker(function
                | Message1 -> printfn "Message1 - Thread Id #%d" Thread.CurrentThread.ManagedThreadId
                | Message2 n -> printfn "Message2 %i - Thread Id #%d" n Thread.CurrentThread.ManagedThreadId
                | Message3 _ -> failwith "I failed"
                , 10 (* 10 agents *)
                , errorHandler = (fun ex _ -> printfn "%A" ex; ContinueProcessing()))


a.Post(Message1)
a.Post(Message3("abc")) // Error
a.Post(Message2(100))
a.Post(Message2(100))


let b = MailboxProcessor.SpawnParallelWorker((fun s -> printfn "%s running on thread %i" s Thread.CurrentThread.ManagedThreadId), 10)
                    

let messages = ['a'..'z'] |> List.map (string)
messages |> Seq.iter (fun msg -> b.Post msg)
   
   

