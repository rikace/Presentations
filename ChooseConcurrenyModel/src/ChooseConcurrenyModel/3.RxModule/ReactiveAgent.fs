module ReactiveAgent

open System
open System.Reactive
open System.Reactive.Subjects
open System.Collections.Generic

 
module ``Reactive Agent`` = 

    type State<'T> = {  observers : IObserver<'T> list
                        stopped   : bool }
            
    type Message<'T> =
    | Add       of IObserver<'T>
    | Remove    of IObserver<'T>
    | Next      of 'T
    | Error     of exn
    | Completed

    type ReactiveAgent<'T>() =
            
        let mbox = MailboxProcessor<Message<'T>>.Start(fun inbox ->
            let rec loop(state:State<'T>) = async {
                if state.stopped then
                    (inbox :> IDisposable).Dispose()
                else
                    let! req = inbox.Receive()
                    match req with
                    | Message.Add(observer) ->                        
                            return! loop ({state with observers = state.observers @ [observer]})                    

                    | Message.Remove(observer) ->                            
                            return! loop {state with observers = state.observers |> List.filter(fun f -> f <> observer)}

                    | Message.Next(value) ->
                            state.observers |> List.iter(fun o -> o.OnNext(value))
                            return! loop state

                    | Message.Error(err) ->
                            state.observers |> List.iter(fun o -> o.OnError(err))
                            return! loop state

                    | Message.Completed ->
                            state.observers |> List.iter(fun o -> o.OnCompleted())                            
                            return! loop {state with stopped = true} }
            loop ({observers = []; stopped=false}) )

        member x.Next value  = mbox.Post <| Message.Next(value)  
        member x.Error ex    = mbox.Post <| Message.Error(ex)    
        member x.Completed() = mbox.Post <| Message.Completed    

        interface IObserver<'T> with
            member x.OnNext value   = x.Next(value)
            member x.OnError ex     = x.Error(ex)
            member x.OnCompleted()  = x.Completed()

        interface IObservable<'T> with
            member x.Subscribe(observer:IObserver<'T>) =
                observer |> Message.Add |> mbox.Post
                { new IDisposable with
                    member x.Dispose() =
                        observer |> Message.Remove |> mbox.Post }





module ``Reactive Agent With LinkTo`` =


    type State<'a> = { isStopped:bool; observers:IObserver<'a> list}
    
    type Message<'a, 'b> =
        | Error of exn
        | Next of 'a * AsyncReplyChannel<'b> option
        | Completed    
        | Subscribe of IObserver<'b>
        | Unsubscribe of IObserver<'b>
        | Stop
 
 
    type AgentRx<'a, 'b>(work:('a -> 'b), ?limit:int, ?token:System.Threading.CancellationToken) =
  
        let token = defaultArg token (new System.Threading.CancellationToken())
        let limit = defaultArg limit 1
        let publish f (observers:IObserver<'b> list) = observers |> List.iter f
        let onCompleted() = publish (fun o -> o.OnCompleted())
        let onError exn = publish (fun o -> o.OnError(exn))
        let onNext item = publish (fun o -> o.OnNext(item))
        let subscribe observer  (observers:IObserver<'b> list) = observer::observers
        let unsubscribe observer (observers:IObserver<'b> list) = observers |> List.filter((<>) observer)
    
        let agent =
            MailboxProcessor<Message<'a, 'b>>.Start(fun inbox ->
 
                        let rec waiting subscribers =
                            inbox.Scan(function 
                                | Completed -> Some( loop (limit - 1) subscribers )
                                | Subscribe(o) -> Some( waiting (subscribe o subscribers) )
                                | Unsubscribe(o) -> Some( waiting (unsubscribe o subscribers) ) 
                                | _ -> None )
                        and loop count subscribers = async {
                            let! msg = inbox.Receive()
                            match msg with
                            | Subscribe(o) -> return! loop count (subscribe o subscribers)
                            | Unsubscribe(o) -> return! loop count (unsubscribe o subscribers)
                            | Stop -> ()
                            | Completed -> return! loop (count - 1) subscribers
                            | Error(exn) -> onError exn subscribers
                                            return! loop count subscribers
                            | Next(item, reply) -> 
                                        async { try
                                                    let res = work item
                                                    match reply with
                                                    | Some(r) -> r.Reply res
                                                    | None -> ()
                                                    onNext res subscribers
                                                finally inbox.Post(Completed) } 
                                        |> Async.Start
                                        if count < limit - 1 then return! loop (count + 1) subscribers
                                        else return! waiting subscribers
                                }
                        loop 0 [])
 
        do
            token.Register(fun () -> agent.Post(Completed)
                                     (agent :> IDisposable).Dispose()) |> ignore
 
        member x.OnNext(item:'a)  = agent.Post(Next(item, None))
        member x.OnCompleted() = agent.Post(Stop)
        member x.OnNextWithResult(item:'a)  = agent.PostAndAsyncReply(fun ch -> Next(item, Some(ch)))

        member x.LinkTo(agentRx:AgentRx<'b, _>) = x.Subscribe(agentRx.OnNext)

        interface IObserver<'a> with
            member x.OnCompleted() = x.OnCompleted()
            member x.OnNext(item) = x.OnNext item
            member x.OnError(exn) = agent.Post(Error exn)
       
        interface IObservable<'b> with
            member x.Subscribe(observer) =
                agent.Post(Subscribe(observer))
                { new IDisposable with
                    member x.Dispose() =
                        agent.Post(Unsubscribe(observer))
                    }
   



    let m = AgentRx(fun n -> printfn "Hello World %s" n
                             7)   

    let d = m.Subscribe(fun n -> printfn "result = %d" (n * n)) 
    m.OnNext("Ciao")


    let n = AgentRx(fun n -> printfn "Get result World %d" n)
    let ref = m.LinkTo(n)

    m.OnNext("ciao")

    (d :> IDisposable).Dispose()
    