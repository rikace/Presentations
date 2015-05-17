namespace BusLib.Actor

open System
open System.Threading

type ErrorMessage = { OriginalMessage:obj; Error:Exception }

exception MessageHandleException of ErrorMessage

type AfterError<'state> =
| ContinueProcessing of 'state
| StopProcessing
    
type MailboxProcessor =

    static member public SpawnAgent<'b>(messageHandler :'a->'b->'b,
                                        initialState : 'b,                                         
                                        ?errorHandler:
                                            Exception -> 'a -> 'b -> AfterError<'b>)
                                        : MailboxProcessor<'a> =
        let errorHandler = defaultArg errorHandler (fun _ _ state -> ContinueProcessing(state))
        MailboxProcessor.Start(fun inbox ->
            let rec loop(state) = async {
                let! msg = inbox.Receive()
                try 
                    return! loop(messageHandler msg state)
                with
                | ex -> match errorHandler ex msg state with
                        | ContinueProcessing(newState)    -> return! loop(newState)
                        | StopProcessing        -> return ()
                }
            loop(initialState))

    static member public SpawnWorker(messageHandler,  ?errorHandler) =
        let errorHandler = defaultArg errorHandler (fun _ _ -> ContinueProcessing(()))
        MailboxProcessor.SpawnAgent((fun msg _ -> messageHandler msg; ()),
                                     (), 
                                     (fun ex msg _ -> errorHandler ex msg))


//type msg1 = Message1 | Message2 of int | Message3 of string
//    let a = MailboxProcessor.SpawnWorker(function
//                    | Message1 -> printfn "Message1";
//                    | Message2 n -> printfn "Message2 %i" n;
//                    | Message3 _ -> failwith "I failed"
//                    , errorHandler = (fun ex _ -> printfn "%A" ex; ContinueProcessing())) // Superviser

[<AutoOpenAttribute; RequireQualifiedAccess>]
module Agent =

    [<SealedAttribute>]
    type AgentRef<'a>(id:string, comp, ?token) = 

        let token = defaultArg token (new CancellationTokenSource())
        let agent = MailboxProcessor.Start((fun inbox -> 
            let rec loop n = async {
                let! msg = inbox.Receive()
                comp(msg)
                return! loop (n + 1)
            }
            loop 0), token.Token)

        member x.Post(msg:'a) = agent.Post(msg)

        member x.PostAndAsyncReply(msg: 'a) = 
            agent.PostAndTryAsyncReply(fun rc -> msg)

        member x.Start() = 
            agent.Start()

        member x.Stop() = 
            token.Cancel()
        
        member x.Id = id
           
    let start (ref:AgentRef<'a>) =
        ref.Start()
        ref

    type Message = 
    | Add of int
    | Get of AsyncReplyChannel<int>

    type AgentRef() =

        
        let agent = MailboxProcessor<Message>.Start(fun inbox ->
                let rec loop n = async {
                    let! msg = inbox.Receive()
                    match msg with
                    | Add(i) -> return! loop (n + i)
                    | Get(r) -> r.Reply(n)
                                return! loop n }
                loop 0)


        member x.SpawnAgent<'b>(messageHandler :'a->'b->'b,
                                initialState : 'b,                                         
                                ?errorHandler:Exception -> 'a -> 'b -> AfterError<'b>) : MailboxProcessor<'a> =
            let errorHandler = defaultArg errorHandler (fun _ _ state -> ContinueProcessing(state))
            MailboxProcessor.Start(fun inbox ->
                let rec loop(state) = async {
                    let! msg = inbox.Receive()
                    try 
                        return! loop(messageHandler msg state)
                    with
                    | ex -> match errorHandler ex msg state with
                            | ContinueProcessing(newState)    -> return! loop(newState)
                            | StopProcessing        -> return ()
                    }
                loop(initialState))

        member x.SpawnWorker(messageHandler,  ?errorHandler) =
            let errorHandler = defaultArg errorHandler (fun _ _ -> ContinueProcessing(()))
            x.SpawnAgent((fun msg _ -> messageHandler msg; ()),
                                     (), 
                                     (fun ex msg _ -> errorHandler ex msg))


    




/// Contains functions to simplify working with MailboxProcessor<> instances
//module Agent =
//    /// Connects error reporting to a supervising MailboxProcessor<>
//    let reportErrorsTo (supervisor: Agent<exn>) (agent: Agent<_>) =
//        agent.Error.Add(fun error -> supervisor.Post error); agent
//
//    /// Starts a MailboxProcessor<> and returns the started instance
//    let start (agent: Agent<_>) = agent.Start(); agent
//    
//    /// Creates a new supervising MailboxProcessor<> from the given error-handling function
//    let supervisor fail = 
//        let processError msg =                                     
//            match msg with
//            | MessageHandleException m -> fail (m.OriginalMessage, m.Error)
//            | exn as Exception -> printfn "An error occurred: Type(%s) Message(%s)" (exn.GetType().Name) exn.Message
//
//        new Agent<exn>(fun inbox ->
//                        let rec Loop() =
//                            async {
//                                let! msg = inbox.Receive()
//                                msg |> processError
//                                do! Loop() }
//                        Loop()) |> start

