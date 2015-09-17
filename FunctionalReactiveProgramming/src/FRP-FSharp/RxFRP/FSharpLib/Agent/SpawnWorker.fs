namespace Easj360FSharp

module SpawnWorker =
    

    type AfterError<'state> =
    | ContinueProcessing of 'state
    | StopProcessing
    | RestartProcessing
    
    type msg = 
    | Increment of int 
    | Fetch of AsyncReplyChannel<int> 
    | Stop

    exception StopException

    type MailboxProcessor<'a> with
        static member public SpawnAgent<'b>(messageHandler :'a->'b->'b,                                        
                                            initialState : 'b, ?timeout:'b -> int,
                                            ?timeoutHandler:'b -> AfterError<'b>,                                        
                                            ?errorHandler: System.Exception -> 'a option -> 'b -> AfterError<'b>)                                        
                                            : MailboxProcessor<'a> =
            let timeout = defaultArg timeout (fun _ -> -1)
            let timeoutHandler = defaultArg timeoutHandler (fun state ->   ContinueProcessing(state))
            let errorHandler = defaultArg errorHandler (fun _ _ state ->   ContinueProcessing(state))
            MailboxProcessor.Start(fun inbox ->
                let rec loop(state) = async {
                    let! msg = inbox.TryReceive(timeout(state))
                    try
                        match msg with
                        | None      -> match timeoutHandler state with
                                        | ContinueProcessing(newState)    ->    return! loop(newState)
                                        | StopProcessing        -> return ()
                                        | RestartProcessing     -> return! loop(initialState)
                        | Some(m)   -> return! loop(messageHandler m state)
                    with
                    | ex -> match errorHandler ex msg state with
                            | ContinueProcessing(newState)    -> return! loop(newState)
                            | StopProcessing        -> return ()
                            | RestartProcessing     -> return! loop(initialState)
                    }
                loop(initialState))
        

        static member public SpawnWorker(messageHandler,  ?timeout, ?timeoutHandler,?errorHandler) =
            let timeout = defaultArg timeout (fun () -> -1)
            let timeoutHandler = defaultArg timeoutHandler (fun _ -> ContinueProcessing(()))
            let errorHandler = defaultArg errorHandler (fun _ _ -> ContinueProcessing(()))
            MailboxProcessor.SpawnAgent((fun msg _ -> messageHandler msg; ()),                                 
                                            (), timeout, timeoutHandler,                                 
                                            (fun ex msg _ -> errorHandler ex msg))

        static member public SpawnParallelWorker(messageHandler, howMany, ?timeout,?timeoutHandler,?errorHandler) =
                    let timeout = defaultArg timeout (fun () -> -1)
                    let timeoutHandler = defaultArg timeoutHandler (fun _ -> ContinueProcessing(()))
                    let errorHandler = defaultArg errorHandler (fun _ _ -> ContinueProcessing(()))
                    MailboxProcessor<'a>.SpawnAgent((fun msg (workers:MailboxProcessor<'a> array, index) ->
                                                        workers.[index].Post msg
                                                        (workers, (index + 1) % howMany))  
                                                    , (Array.init howMany                                      
                                                    (fun _ -> MailboxProcessor<'a>.SpawnWorker(messageHandler, timeout, timeoutHandler,                                                                                         errorHandler)), 0))

                (*
•messageHandler: a function to execute when a message comes in, it takes the message and the current state as parameters and returns the new state.
•initialState: the initial state for the MailboxProcessor
•timeoutHandler: a function that is executed whenever a timeout occurs. It takes as a parameter the current state and returns one of ContinueProcessing 
                (newState), StopProcessing or RestartProcessing
•errorHandler: a function that gets call if an exception is generated inside the messageHandler function. It takes the exception, the message, the current state                  and returns ContinueProcessing(newState), StopProcessing or RestartProcessing
                *)

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
        
