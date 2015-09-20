namespace FsBus

open System
open System.Messaging
open System.Transactions
open System.Collections.Generic

type CreateGuitarCommand() = 
    let mutable name = ""
    member x.Name with get() = name and set v = name <- v

type DeleteGuitarCommand() = 
    let mutable name = ""
    member x.Name with get() = name and set v = name <- v
    member x.RequestData with get() = DateTime.Now

type IMessageBus =
    abstract member QueueName : string
    abstract member Publish : 'a -> unit
    abstract member Subscribe<'a> : Action<'a> -> Action<Exception, obj> -> unit
    inherit IDisposable 

type MessageBusImplDetails =
    { QueueName : string
      IsTransactionalQueue : bool
      MessageQueueInstance : MessageQueue
      IsRemoteQueue : bool 
      Publisher : MessageQueue -> bool -> Message -> unit
      IsForTesting : bool }

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

type MessageBus(implDetails) =    
    let messageQueue = implDetails.MessageQueueInstance
    new (queueName, isTransactionQueue) = new MessageBus( GetMessageBusImplDetails(queueName, isTransactionQueue) )
    interface IMessageBus with
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

type Consumer() =
    //let bus = new MessageBus("sample_queue2", false)

//    [1..1000000] |> Seq.iter (fun x -> 
//                        bus.Publish (new DeleteGuitarCommand(Name="test" + x.ToString()))
//                        printfn "Publishing message %i" x)

//    printfn "Press any key to quite\r\n"
//    Console.ReadLine() |> ignore
//    bus.Dispose()

    //printfn "Waiting for a message"

    //let createGuitarCommands = new MessageBus("sample_queue")
    let deleteGuitarCommands = new MessageBus("sample_queue", false)

    //createGuitarCommands.Subscribe<CreateGuitarCommand> 
    //                        (new Action<_>(fun cmd -> printfn "A request for a new Guitar with name %s was consumed" cmd.Name))
    do
      deleteGuitarCommands.Subscribe<obj>
                            (new Action<_>(fun cmd -> printfn "A request to DELETE a Guitar with name %A was consumed" cmd))
                            (new Action<Exception, obj>(fun ex o -> 
                                printfn "Exception: %s and message: %s" ex.Message (o.ToString())))
    
//    printfn "Press any key to quite\r\n"
//    Console.ReadLine() |> ignore
//    //createGuitarCommands.Dispose()
//    deleteGuitarCommands.Dispose()

type Producer() = 
    let bus = new MessageBus("FormatName:Direct=OS:localhost\\private$\\test", true)

    let test = 
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

        printfn "Press any key to quite\r\n"
        Console.ReadLine() |> ignore
        bus.Dispose()