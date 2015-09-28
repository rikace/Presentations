module SuperBus

open System
open System.Collections.Generic
open System.Threading
open System.Threading.Tasks
open System.Reactive.Linq

type Subject<'a> = Reactive.Subjects.Subject<'a>


// Message bus used to distribute messages.
type IBus<'a> =
    inherit IObservable<'a>
    inherit IDisposable 
    
    // Adds the given <see cref="IObservable{T}"/> as a message source.    
    abstract AddPublisher : IObservable<'a> -> unit

  // Subscribes an action to the message bus.
type ISubscriptionService<'a> =
    // Subscribes the given handler to the message bus. Only messages for which the given predicate resolves to true will be passed to the handler.
    abstract Subscribe : ('a -> bool) -> ('a -> unit) -> IDisposable 
  


// Implementation of <see cref="IBus"/> that keeps publishers and subscriptions in memory.
[<SealedAttribute>]
type InMemoryBus<'a>() = // : IBus {
    let subject = new Subject<'a>()
    let publisherSubscriptions = new List<IDisposable>()

    interface IBus<'a> with 
        // Adds the given <see cref="IObservable{T}"/> as a message source.
        member x.AddPublisher(observable:IObservable<'a>) =
          //if (observable == null) throw new ArgumentNullException("observable");
          publisherSubscriptions.Add(observable.Subscribe(fun msg -> subject.OnNext(msg)))
    

        // Notifies the provider that an observer is to receive notifications.
        // A reference to an interface that allows observers to stop receiving notifications before the provider has finished sending them.
        member x.Subscribe(observer:IObserver<'a>) : IDisposable =
          subject.Subscribe(observer)


        member x.Dispose() =
            publisherSubscriptions.ForEach(fun d -> d.Dispose())
            subject.Dispose()

// Default implementation of <see cref="ISubscriptionService"/>.
[<SealedAttribute>]
type SubscriptionService<'a>(observable:IObservable<'a>) =
    // Creates a new instance of <see cref="SubscriptionService"/>.
    
    interface ISubscriptionService<'a> with
    // Subscribes the given handler to the message bus. Only messages for which the given predicate resolves to true will be passed to the handler.
        member x.Subscribe canHandle handle: IDisposable =            
                observable.Where(canHandle).Subscribe(handle) 

            //.OfType<'a>().Where(msg => canHandle(msg)).Subscribe(handle);
    
       


(*
let bus = new MessageBus("FormatName:Direct=OS:localhost\\private$\\test", true)

bus.Publish (new DeleteGuitarCommand(Name="test"))
printfn "Publishing delete message 1"

bus.Publish (new CreateGuitarCommand(Name="test"))
printfn "Publishing message 1"

bus.Publish (new DeleteGuitarCommand(Name="test"))
printfn "Publishing delete message 2"

bus.Publish (new CreateGuitarCommand(Name="test2"))
printfn "Publishing message 2"

bus.Publish (new DeleteGuitarCommand(Name="test"))
printfn "Publishing delete message 3"

bus.Publish (new CreateGuitarCommand(Name="test3"))
printfn "Publishing message 3"

bus.Publish (new DeleteGuitarCommand(Name="test"))
printfn "Publishing delete message 3"

//[1..1000000] |> Seq.iter (fun x -> 
//                    bus.Publish (new CreateGuitarCommand(Name="test" + x.ToString()))
//                    printfn "Publishing message %i" x)
bus.Subscribe(fun f -> printfn "%A" f)
printfn "Press any key to quite\r\n"
Console.ReadLine() |> ignore
bus.Dispose()






type CreateGuitarCommand() = 
    let mutable name = ""
    member x.Name with get() = name and set v = name <- v

type DeleteGuitarCommand() = 
    let mutable name = ""
    member x.Name with get() = name and set v = name <- v
    member x.RequestData with get() = DateTime.Now


open Messages
open System
open FsBus

printfn "Waiting for a message"

//let createGuitarCommands = new MessageBus("sample_queue")
let deleteGuitarCommands = new MessageBus("sample_queue", false)

//createGuitarCommands.Subscribe<CreateGuitarCommand> 
//                        (new Action<_>(fun cmd -> printfn "A request for a new Guitar with name %s was consumed" cmd.Name))

deleteGuitarCommands.Subscribe<obj>
                        (new Action<_>(fun cmd -> printfn "A request to DELETE a Guitar with name %A was consumed" cmd))
                        (new Action<Exception, obj>(fun ex o -> 
                            printfn "Exception: %s and message: %s" ex.Message (o.ToString())))
    
printfn "Press any key to quite\r\n"
Console.ReadLine() |> ignore
//createGuitarCommands.Dispose()
deleteGuitarCommands.Dispose()



let ``When publishing two message types and consuming them should have expected results`` () =
    try
        let result1 = ref ""
        let result2 = ref ""
        publisher1.Publish (new DeleteGuitarCommand(Name="test"))
        publisher2.Publish (new CreateGuitarCommand(Name="test2"))

        subscriber1.Subscribe<DeleteGuitarCommand>
                        (new Action<_>(fun cmd -> result1 := cmd.Name))
                        (new Action<Exception, obj>(fun ex o -> 
                            result1 := ex.Message
                            printfn "Exception: %s and message: %s" ex.Message (o.ToString())))
        subscriber2.Subscribe<CreateGuitarCommand>
                        (new Action<_>(fun cmd -> result2 := cmd.Name))
                        (new Action<Exception, obj>(fun ex o -> 
                            result2 := ex.Message
                            printfn "Exception: %s and message: %s" ex.Message (o.ToString())))
        let rec loop () =
            match !result1 = "" && !result2 = "" with
            | false -> 
                !result1 |> should equal "test"
                printfn "Result for 1 - %s" !result1 
                !result2 |> should equal "test2"
                printfn "Result for 2 - %s" !result2
            | true -> loop ()
        loop()
        ()
    finally
        publisher1.Dispose()
        publisher2.Dispose()
        subscriber1.Dispose()
        subscriber2.Dispose()


//
//type MessageBusImplDetails =
//    { QueueName : string
//      IsTransactionalQueue : bool
//      MessageQueueInstance : MessageQueue
//      IsRemoteQueue : bool 
//      Publisher : MessageQueue -> bool -> Message -> unit
//      IsForTesting : bool }

[<AutoOpen>]
module MessageBusImplMod =
    let createQueueIfMissing (queueName:string) =
        if not (MessageQueue.Exists queueName) then
            MessageQueue.Create(queueName, true) |> ignore

    let parseQueueName (queueName:string) =             
        match queueName with
        | _ when queueName.Contains("@") && queueName.Split('@').[1] <> "localhost" -> 
            true, queueName.Split('@').[1] + "\\private$\\" + queueName.Split('@').[0]
        | _ when queueName.Contains("\\") -> true, queueName
        | _ -> false, ".\\private$\\" + queueName        
    
    let getMessageQueueInfo queueName = 
        let isRemote, qName = parseQueueName queueName
        if not isRemote then createQueueIfMissing qName
        new MessageQueue(qName), isRemote

    let SendMessageToQueue (messageQueue:MessageQueue) isTransactionalQueue (msg:Message) = 
        isTransactionalQueue
        |> function
           | true -> 
               use scope = new TransactionScope()
               messageQueue.Send(msg, MessageQueueTransactionType.Automatic)
               scope.Complete()
           | _ -> messageQueue.Send(msg)        

    let GetMessageBusImplDetails (queueName:string, isTransactionalQueue:bool) =
        let messageQueue, isRemoteQueue = getMessageQueueInfo queueName
        let isTransQueue = match isRemoteQueue with
                           | true -> isTransactionalQueue
                           | _ -> messageQueue.Transactional
        { QueueName = queueName; IsTransactionalQueue = isTransQueue; MessageQueueInstance = messageQueue; IsRemoteQueue = isRemoteQueue
          Publisher = SendMessageToQueue; IsForTesting = false }

type MessageBus() =    
    
        member x.QueueName with get () = implDetails.QueueName
        
        member x.Publish message = 
            let msgTypeName = message.GetType().AssemblyQualifiedName
            new Message(Label = msgTypeName, Body = message, Formatter = new BinaryMessageFormatter())
            |> implDetails.Publisher messageQueue implDetails.IsTransactionalQueue  
        
        member x.Subscribe<'a> (success:Action<'a>) (failure:Action<Exception, obj>) =     
            match implDetails.IsForTesting with
            | true -> ()
            | _ -> 
                messageQueue.ReceiveCompleted.Add( 
                    fun (args) -> 
                        try                              
                            args.Message.Formatter <- new BinaryMessageFormatter()
                            args.Message.Body :?> 'a |> success.Invoke
                        with
                        | ex ->
                            try
                              failure.Invoke(ex, args.Message.Body)  
                            with
                            | ex -> raise ex
                        messageQueue.BeginReceive() |> ignore)
                messageQueue.BeginReceive() |> ignore

        member x.Dispose () = messageQueue.Dispose()
    member x.QueueName with get() = (x :> IMessageBus).QueueName
    member x.Publish message = (x :> IMessageBus).Publish message    
    member x.Subscribe<'a> (success:Action<'a>) (failure:Action<Exception, obj>) = (x :> IMessageBus).Subscribe<'a> success failure
    member x.Dispose() = (x :> IMessageBus).Dispose()


    *)