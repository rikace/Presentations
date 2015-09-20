open System
open System.Collections
open System.Threading

type ObservableMsg =
    | Subscribe of IObserver<Message>
    | Unsubscribe of IObserver<Message>
    | Message of Message

and Message =
    | Add of int
    | Print of string
    | Error of exn

type obsAgent () =

    let agent = MailboxProcessor<Message>.Start(fun inbox ->
                let rec loop n = async{
                    let! msg = inbox.Receive()
                    match msg with
                    | Add(i) -> printfn "Message N %d - Nummebr Add - %d" n ( n + i)
                    | Print(s) -> printf "Print Message %d - string -> %s" n s
                    return! loop (n + 1) }
                loop 0)

    interface  IObserver<Message> with
            member x.OnNext(value) = 
                agent.Post value
            member x.OnError(exn) =
                agent.Post (Error(exn))
            member x.OnCompleted() =
                () 
        
 
type observableAgent() =
    let agent = MailboxProcessor<ObservableMsg>.Start(fun inbox ->
                let rec loop n (observers:IObserver<Message> list) = async{
                    let! msg = inbox.Receive()
                    match msg with
                    | Subscribe(o) -> 
                            return! loop (n + 1) (o::observers)
                    | Unsubscribe(o) -> return! loop (n + 1) (observers |> List.filter(fun f -> f <> o))
                    | Message(s) -> observers |> List.iter (fun o -> o.OnNext s)
                                    return! loop(n + 1) observers

                }
                loop 0 [])


    let obs = { new IObservable<Message> with
                    member x.Subscribe(obs) = 
                        agent.Post(Subscribe obs)
                        { new IDisposable with
                               member this.Dispose() =
                                agent.Post(Unsubscribe obs)
                        }
                     }

    member x.Notify(msg) =
        agent.Post(Message msg)  

    member this.AsObservable = obs



[<EntryPoint>]
let main argv = 
    
    let obse = new observableAgent() //:> IObservable<string>
    let obs1 = new obsAgent()
    let obs2 = obsAgent()
   
    let disp1 = obse.AsObservable.Subscribe(obs1)
    let disp2 = obse.AsObservable.Subscribe(obs2)


    obse.Notify(Add(7))
    obse.Notify(Print("Ciao"))
    disp1.Dispose()
   

    Console.ReadLine() |> ignore

    0 // return an integer exit code
