#r "..\..\Lib\ActorLib.dll"

open System
open FSharp.Actor

(************************* Calculator *****************************)

//  Basic Actors
let multiplication = 
    (fun (actor:IActor<_>) ->
        let rec loop() =
            async {
                let! ((a,b), sender) = actor.Receive()
                let result = a * b
                do printfn "%A: %d * %d = %d" actor.Path a b result
                return! loop()
            }
        loop()
    )

let addition = 
    (fun (actor:IActor<_>) ->
        let rec loop() =
            async {
                let! ((a,b), sender) = actor.Receive()
                let result = a + b
                do printfn "%A: %d + %d = %d" actor.Path a b result
                return! loop()
            }
        loop()
    )

let calculator = 
    [
       Actor.spawn (Actor.Options.Create("calculator/addition")) addition
       Actor.spawn (Actor.Options.Create("calculator/multiplication")) multiplication
    ]
//  The above code creates two actors calcualtor/addition and calculator/multiplication

"calculator/addition" ?<-- (5,2)
"calculator/multiplication" ?<-- (5,2)

calculator.[0] <-- (5,2)

//  Or we could have broadcast to all of the actors in that collection
calculator <-* (5,2)

// We can also resolve systems of actors.
"calculator" ?<-- (5,2)

//  However this actor wont be found because it does not exist
"calculator/addition/foo" ?<-- (5,2)

calculator.[1] <!- (Shutdown("Cause I want to"))
//  or
"calculator/addition" ?<!- (Shutdown("Cause I want to"))

let rec schizoPing = 
    (fun (actor:IActor<_>) ->
        let log = (actor :?> Actor.T<_>).Log
        let rec ping() = 
            async {
                let! (msg,_) = actor.Receive()
                log.Info(sprintf "(%A): %A ping" actor msg, None)
                return! pong()
            }
        and pong() =
            async {
                let! (msg,_) = actor.Receive()
                log.Info(sprintf "(%A): %A pong" actor msg, None)
                return! ping()
            }
        ping()
    )
        

let schizo = Actor.spawn (Actor.Options.Create("schizo")) schizoPing 
!!"schizo" <-- "Hello"
//Sending two messages to the 'schizo' actor results in


//  Linking Actors
//  Linking an actor to another means that this actor will become a
//  sibling of the other actor. This means that we can create relationships among actors
let child i = 
    Actor.spawn (Actor.Options.Create(sprintf "a/child_%d" i)) 
         (fun actor ->
             let log = (actor :?> Actor.T<_>).Log 
             let rec loop() =
                async { 
                   let! msg = actor.Receive()
                   log.Info(sprintf "%A recieved %A" actor msg, None) 
                   return! loop()
                }
             loop()
         )

let parent = 
    Actor.spawnLinked (Actor.Options.Create "a/parent") (List.init 5 (child))
            (fun actor -> 
                let rec loop() =
                  async { 
                      let! msg = actor.Receive()
                      actor.Children <-* msg
                      return! loop()
                  }
                loop()    
            ) 

parent <-- "Forward this to your children"


//  We can also unlink actors
Actor.unlink !*"a/child_0" parent

parent <-- "Forward this to your children"

//  State in Actors
//  State in actors is managed by passing an extra parameter around the loops. For example,
let incrementer =
    Actor.spawn Actor.Options.Default (fun actor -> 
        let log = (actor :?> Actor.T<int>).Log
        let rec loopWithState (currentCount:int) = 
            async {
                let! (a,_) = actor.Receive()
                log.Debug(sprintf "Incremented count by %d" a, None) 
                return! loopWithState (currentCount + a)
            }
        loopWithState 0
    )

incrementer <-- 1
incrementer <-- 2

//  However the if the actor dies this state is lost. We need a way of rebuilding this state.
//  Here we can use event sourcing. We can persist the events as they pour into the actor
//  then on restart replay those events.
//type Messages = 
//    | Incr of int 
//    | Seed of int list
//
//let eventSourcedIncrementer (eventStore:IEventStore) =
//    Actor.spawn Actor.Options.Default (fun actor -> 
//        let log = (actor :?> Actor.T<Messages>).Log
//        let rec loopWithState (currentCount:int) = 
//            async {
//                let! (a,_) = actor.Receive()
//                match a with
//                | Incr a ->
//                    log.Debug(sprintf "Incremented count by %d" a, None)
//                    let newState = currentCount + a
//                    eventStore.Store(actor.Id, a)
//                    log.Debug (sprintf "Current state %d" newState, None)
//                    return! loopWithState newState
//                | Seed a -> 
//                    return! loopWithState (a |> List.fold (+) currentCount)    
//            }
//        loopWithState 0
//    )
//
//let eventStore = new InMemoryEventStore() :> IEventStore
//
//let pIncrementer = eventSourcedIncrementer eventStore
//pIncrementer.OnRestarted |> Event.add (fun actor -> 
//                                        let events = 
//                                            eventStore.Replay(actor.Id) 
//                                            |> Async.RunSynchronously 
//                                            |> Seq.toList
//                                        actor <-- Seed events )
//
//pIncrementer <-- Incr 1
//pIncrementer <-- Incr 2
//
//pIncrementer.PostSystemMessage(SystemMessage.Restart("Just testing seeding"), None)
//
//pIncrementer <-- Incr 3

(*  Above we are passing in a event store to store the incremental changes to the actor. 
    We then subscribe the OnRestarted event. This provides us with a hook then query the
    event store and replay the events to build up the Message to reseed the actor.  *)




// Remoting
(*  Remoting can be used to send messages to connected actor systems outside of 
    the current process boundary. To use remoting we must first register a transport
    and a serialiser to use with that transport. Transports are responsible for
    packaging, sending and recieving messages to/from other remote systems. 
    Each transport should have a scheme thast uniquely identifies it. To register 
    a transport, do something like the following *)
let transport = new Fracture.FractureTransport(8080) :> ITransport
Registry.Transport.register transport

(*  The above call registers a transport that uses Fracture. When a transport is 
    created it is wrapped in a actor of Actor<RemoteMessage> and path 
    'transports/{scheme}' in the case of the fracture transport this would be
    'transports/actor.fracture'. This actor is then supervised by the system/remotingsupervisor
    actor, which is initialised the a OneForOne strategy so will restart the transport
    if it errors at any point.

    Sending a message to a remote actor is identical to sending messages to local actors
    apart from the actor path has to be fully qualified. *)
"actor.fracture://127.0.0.1:8081/RemoteActor" ?<-- "Some remote actor message"

(*  In addition to sending normal messages to a remote actor, we can also 
    send system messages. For example if we want to restart a remote actor
    We could send the following message *)
"actor.fracture://127.0.0.1:8081/RemoteActor" ?<-- Restart

//  Round robin
//  Round robin dispatch, distributes messages in a round robin fashion to its workers.
let createWorker i =
    Actor.spawn (Actor.Options.Create(sprintf "workers/worker_%d" i)) (fun (actor:IActor<int>) ->
        let log = (actor :?> Actor.T<int>).Log
        let rec loop() = 
            async {
                let! (msg,_) = actor.Receive()
                do log.Debug(sprintf "Actor %A recieved work %d" actor msg, None)
                do! Async.Sleep(5000)
                do log.Debug(sprintf "Actor %A finshed work %d" actor msg, None)
                return! loop()
            }
        loop()
    )

let workers = [|1..10|] |> Array.map createWorker
let rrrouter = Patterns.Dispatch.roundRobin<int> "workers/routers/roundRobin" workers

[1..10] |> List.iter ((<--) rrrouter)

//  Shortest Queue
(*  Shortest queue, attempts to find the worker with the shortest queue and 
    distributes work to them. For constant time work packs this will approximate
    to round robin routing.

    Using the workers defined above we can define another dispatcher but this 
    time using the shortest queue dispatch strategy *)

let sqrouter = Patterns.Dispatch.shortestQueue "workers/routers/shortestQ" workers

[1..100] |> List.iter ((<--) sqrouter)


//  Supervising Actors
//  Actors can supervise other actors, if we define an actor loop that fails on a given message
let err = 
        (fun (actor:IActor<string>) ->
            let rec loop() =
                async {
                    let! (msg,_) = actor.Receive()
                    if msg <> "fail"
                    then printfn "%s" msg
                    else failwithf "ERRRROROROR"
                    return! loop()
                }
            loop()
        )
//  then a supervisor will allow the actor to restart or terminate depending on 
//  the particular strategy that is in place

//  Strategies
//  A supervisor strategy allows you to define the restart semantics for the actors it is watching

//  OneForOne
//  A supervisor will only restart the actor that has errored
let oneforone = 
    Supervisor.spawn 
        <| Supervisor.Options.Create(actorOptions = Actor.Options.Create("OneForOne"))
    |> Supervisor.superviseAll [Actor.spawn (Actor.Options.Create("err_0")) err]

!!"err_0" <-- "fail"

//  OneForAll
//  If any watched actor errors all children of this supervisor will be told to restart.
let oneforall = 
    Supervisor.spawn 
        <| Supervisor.Options.Create(
                    strategy = Supervisor.Strategy.OneForAll,
                    actorOptions = Actor.Options.Create("OneForAll")
           )
    |> Supervisor.superviseAll
        [
            Actor.spawn (Actor.Options.Create("err_1")) err;
            Actor.spawn (Actor.Options.Create("err_2")) err
        ]
"err_1" ?<-- "Boo"
"err_2" ?<-- "fail"

//  Fail
//  A supervisor will terminate the actor that has errored
let fail = 
    Supervisor.spawn 
        <| Supervisor.Options.Create(
                    strategy = Supervisor.Strategy.AlwaysFail,
                    actorOptions = Actor.Options.Create("Fail")
           )
    |> Supervisor.superviseAll
        [
            Actor.spawn (Actor.Options.Create("err_3")) err;
            Actor.spawn (Actor.Options.Create("err_4")) err
        ]

!!"err_3" <-- "fail"

let oneforallunwatch = 
    Supervisor.spawn 
        <| Supervisor.Options.Create(
                    strategy = Supervisor.Strategy.OneForAll,
                    actorOptions = Actor.Options.Create("OneForAll")
           )
    |> Supervisor.superviseAll
        [
            Actor.spawn (Actor.Options.Create("err_5")) err;
            Actor.spawn (Actor.Options.Create("err_6")) err
        ]

Actor.unwatch !*"err_6" 
!!"err_5" <-- "fail"


(************************* NODE *****************************)
let fractureTransport port = 
    new Fracture.FractureTransport(port)

let logger node = 
    Actor.spawn (Actor.Options.Create(node)) 
       (fun (actor:IActor<string>) ->
            let log = (actor :?> Actor.T<string>).Log
            let rec loop() = 
                async {
                    let! (msg, sender) = actor.Receive()
                    log.Debug(sprintf "%A sent %A" sender msg, None)
                    match sender with
                    | Some(s) -> 
                        s <-- "pong"
                    | None -> ()
                    return! loop()
                }
            loop()
        )


let logSamplev = 
    let loggerNode1 = logger "node1/logger"
    let loggerNode2 = logger "node2/logger"

    Registry.Transport.register (fractureTransport 6667)
    //Registry.Transport.register (fractureTransport 6666)
    
    loggerNode1 <-- "Hello"
    "node1/logger" ?<-- "Hello"

    while Console.ReadLine() <> "exit" do
        "actor.fracture://127.0.0.1:6666/node2/logger" ?<-- "Ping"
        "actor.fracture://127.0.0.1:6667/node1/logger" ?<-- "Ping"

    "actor.fracture://127.0.0.1:6666/node2/logger" ?<!- Shutdown("Remote Shutdown")
    "actor.fracture://127.0.0.1:6667/node1/logger" ?<!- Shutdown("Remote Shutdown")
    

(************************* Broadcast Actor *****************************)
type BroadcastActor<'a>(actorName: string) =
    let broadcastees = ref Set.empty<string>  

    let broadcaster = 
        Actor.spawn(Actor.Options.Create(actorName, ?logger = Some Logging.Silent))
            (fun (actor: IActor<'a>) ->
                let rec loop() =
                    async {
                        let! (msg, sender) = actor.Receive()
                        !broadcastees
                        |> Seq.iter(fun i -> i ?<-- msg)
                        return! loop()
                    }
                loop()
            )

    member x.AddSubscriber(remotePath) =
        broadcastees := (!broadcastees).Add(remotePath)

    member x.RemoveSubscriber(remotePath) =
        broadcastees := (!broadcastees).Remove(remotePath)

    member x.Agent =
        broadcaster

(************************* SURGE CLIENT *****************************)
open System
open System.Configuration
open System.Drawing
open System.Windows.Forms
open FSharp.Actor
open FSharp.Actor.Surge

type RequestMessage =
    | Connect
    | Subscribe of string
    | Unsubscribe of string

type RequestWithPath = {
    Request: RequestMessage
    ResponsePath: string
}

let surgeTransportClient =  new SurgeTransport(1337)

let serverAddress = "127.0.0.1:1338"
let clientAddress = "127.0.0.1:1337"

let subscribe(product) =
    let subscriptionPath = sprintf "subscriptions/%s/%s" product (Guid.NewGuid().ToString())
        
    Async.Start(
        async {
            let form = new Form(Width = 225, Height = 100, Text = product)
            let context = System.Threading.SynchronizationContext.Current
            let label = new Label()
            form.Controls.Add(label)

            let updateLabel(text: string) =
                async {
                    do! Async.SwitchToContext(context)
                    label.Text <- text
                }

            let receiver = 
                Actor.spawn (Actor.Options.Create(subscriptionPath, ?logger = Some Logging.Silent))
                    (fun (actor:IActor<string>) ->
                        let rec loop() =
                            async {
                                let! (msg, sender) = actor.Receive()
                                do! updateLabel(msg)
                                return! loop()
                            }
                        loop()
                    )

            receiver.OnStopped.Add(fun e ->
                Console.WriteLine(sprintf "Disposing client subscription to %s" product)
            )

            form.Closed.Add(fun e ->
                do sprintf "actor.surge://%s/server/subscriptions" serverAddress ?<-- { Request = Unsubscribe(product); ResponsePath = sprintf "actor.surge://%s/%s" clientAddress subscriptionPath }
                receiver.PostSystemMessage(Shutdown("Disposing subscription"), None)
            )

            Application.Run(form)
        }
    )

    do sprintf "actor.surge://%s/server/subscriptions" serverAddress ?<-- { Request = Subscribe(product); ResponsePath = sprintf "actor.surge://%s/%s" clientAddress subscriptionPath }
    
let startClient = 
    Registry.Transport.register surgeTransportClient

    while true do
        Console.WriteLine("\nEnter the name of a stock symbol to subscribe to quote updates: ")
        let symbol = Console.ReadLine()
        subscribe(symbol) |> ignore
    


(************************* SURGE SERVER *****************************)
open System
open System.Collections.Generic
open FSharp.Actor
open FSharp.Actor.Surge

let surgeTransportServer = new SurgeTransport(1338)

let createQuoteProducer(product: string, broadcaster: IActor) = 
    async {
        let rec produceQuote(count: int) =
            do Async.Sleep(250) |> Async.RunSynchronously
            broadcaster.Post(sprintf "%s Quote %u" product count, None)
            produceQuote(count + 1)
        produceQuote(0)
    }

let createQuoteBroadcaster(product) =
    Actor.spawn (Actor.Options.Create(sprintf "server/subscriptions/broadcasters/%s" product, ?logger = Some Logging.Silent))
        (fun (actor:IActor<string>) ->
            let rec loop() =
                async {
                    let! (msg, sender) = actor.Receive()
                    actor.Children <-* msg
                    return! loop()
                }
            loop()
        )

let createQuoteSubscription(product, responsePath) =
    Actor.spawn (Actor.Options.Create(sprintf "server/subscriptions/%s/%s" product responsePath, ?logger = Some Logging.Silent))
        (fun (actor:IActor<string>) ->
            let rec loop() =
                async {
                    let! (msg, sender) = actor.Receive()
                    responsePath ?<-- msg
                    return! loop()
                }
            loop()
        )

let subscriptionManager =
    let quoteBroadcasters = new Dictionary<string, IActor>()

    Actor.spawn (Actor.Options.Create("server/subscriptions", ?logger = Some Logging.Silent))
        (fun (actor:IActor<RequestWithPath>) ->
            let rec loop() =
                async {
                    let! (msg, sender) = actor.Receive()
                    match msg.Request with
                    
                    | Subscribe(product) -> 
                        Console.WriteLine(sprintf "\nAdding subscriber %s for %s quote stream" msg.ResponsePath product) 
                        if quoteBroadcasters.ContainsKey(product) then
                            match quoteBroadcasters.TryGetValue(product) with
                            | (true, broadcaster) -> 
                                broadcaster.Link (createQuoteSubscription(product, msg.ResponsePath))
                            | _ -> Console.WriteLine("error")
                        else
                            let broadcaster = createQuoteBroadcaster(product)
                            broadcaster.Link(createQuoteSubscription(product, msg.ResponsePath))
                            quoteBroadcasters.Add(product, broadcaster) 
                            createQuoteProducer(product, broadcaster) |> Async.Start

                    | Unsubscribe(product) ->
                         if quoteBroadcasters.ContainsKey(product) then
                            match quoteBroadcasters.TryGetValue(product) with
                            | (true, broadcaster) ->
                                Console.WriteLine(sprintf "\nRemoving subscriber %s from %s quote stream" msg.ResponsePath product) 
                                broadcaster.UnLink !! (sprintf "server/subscriptions/%s/%s" product msg.ResponsePath) 
                            | _ -> Console.WriteLine("error")
                    
                    | _ -> msg.ResponsePath ?<-- "Other message type"
                    
                    return! loop();
                }
            loop()
        )

let StartSerever =
    Registry.Transport.register surgeTransportServer
    
