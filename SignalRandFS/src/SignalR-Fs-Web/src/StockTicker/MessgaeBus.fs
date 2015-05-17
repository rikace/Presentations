namespace StockTicker.MessageBus

open System
open System.Messaging
open System.Transactions
open System.Collections.Generic

//module SimpleBus =
//
//    open System
//    open System.Messaging
//    open System.Threading
//
//    let private createQueueIfMissing (queueName:string) =
//        if not (MessageQueue.Exists queueName) then
//            MessageQueue.Create queueName |> ignore
//
//    let private parseQueueName (queueName:string) =             
//        let fullName = match queueName.Contains("@") with
//                       | true when queueName.Split('@').[1] <> "localhost" -> 
//                           queueName.Split('@').[1] + "\\private$\\" + 
//                               queueName.Split('@').[0]
//                       | _ -> ".\\private$\\" + queueName
//        createQueueIfMissing fullName
//        fullName
//
//    let subscribe<'a> queueName success failure =     
//        let queue = new MessageQueue(parseQueueName queueName)
//
//        queue.ReceiveCompleted.Add( 
//            fun (args) ->     
//                try        
//                    args.Message.Formatter <- new BinaryMessageFormatter() 
//                    args.Message.Body :?> 'a |> success
//                with
//                | ex -> failure ex args.Message.Body
//                queue.BeginReceive() |> ignore)
//
//        queue.BeginReceive() |> ignore
//        queue
//
//    let publish queueName message =     
//        use queue = new MessageQueue(parseQueueName queueName)
//        new Message(message, new BinaryMessageFormatter())
//        |> queue.Send


type CreateGuitarCommand = { 
    Name : string 
    }

//module PublishMonad
//
//// Define the SendMessageWith Discriminated Union
//type SendMessageWith<'a> = SendMessageWith of string * 'a
//
//// Define the PublishBuilder builder type
//type PublishBuilder() =
//    member x.Bind(SendMessageWith(q, msg):SendMessageWith<_>, fn) =
//        // Note: This is for illustration purposes
//        //       A "real" trace would log to a file, DB, event log, etc.
//        printfn "A message was sent to queue %s" q        
//        SimpleBus.publish q msg
//    member x.Return(_) = true
//
//// Create the builder-name   
//let publish = new PublishBuilder()
//
//namespace Messages
//
////type CreateGuitarCommand() = 
////    member val Name = "" with get, set
//
//type CreateGuitarCommand = { 
//    Name : string 
//    }
//
//
//module SimpleBus
//
//open System
//open System.Messaging
//
//let private createQueueIfMissing (queueName:string) =
//    if not (MessageQueue.Exists queueName) then
//        MessageQueue.Create queueName |> ignore
//
//let private parseQueueName (queueName:string) =             
//    let fullName = match queueName.Contains("@") with
//                   | true when queueName.Split('@').[1] <> "localhost" -> 
//                       queueName.Split('@').[1] + "\\private$\\" + 
//                           queueName.Split('@').[0]
//                   | _ -> ".\\private$\\" + queueName
//    createQueueIfMissing fullName
//    fullName
//
//let subscribe<'a> queueName success failure =     
//    let queue = new MessageQueue(parseQueueName queueName)
//
//    queue.ReceiveCompleted.Add( 
//        fun (args) ->     
//            try        
//                args.Message.Formatter <- new BinaryMessageFormatter() 
//                args.Message.Body :?> 'a |> success
//            with
//            | ex -> failure ex args.Message.Body
//            queue.BeginReceive() |> ignore)
//
//    queue.BeginReceive() |> ignore
//    queue
//
//let publish queueName message =     
//    use queue = new MessageQueue(parseQueueName queueName)
//    new Message(message, new BinaryMessageFormatter())
//    |> queue.Send
//
//open Messages
//open System
//
//printfn "Waiting for a message"
//
//let queueToDispose = 
//    SimpleBus.subscribe<CreateGuitarCommand> "sample_queue" 
//        (fun cmd -> 
//            printfn "A message for a new guitar named %s was consumed" cmd.Name)
//        (fun (ex:Exception) o -> 
//            printfn "An exception occurred with message %s" ex.Message)     
//    
//printfn "Press any key to quite\r\n"
//Console.ReadLine() |> ignore
//queueToDispose.Dispose()




//namespace Messages

//type CreateGuitarCommand() = 
//    member val Name = "" with get, set
//
//type CreateGuitarCommand = { 
//    Name : string 
//    }


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

//[<AutoOpen>]
//module MessageBusImplMod =
//    let createQueueIfMissing (queueName:string) =
//        if not (MessageQueue.Exists queueName) then
//            MessageQueue.Create(queueName, true) |> ignore
//
//    let parseQueueName (queueName:string) =             
//        match queueName with
//        | _ when queueName.Contains("@") && queueName.Split('@').[1] <> "localhost" -> 
//            true, queueName.Split('@').[1] + "\\private$\\" + queueName.Split('@').[0]
//        | _ when queueName.Contains("\\") -> true, queueName
//        | _ -> false, ".\\private$\\" + queueName        
//    
//    let getMessageQueueInfo queueName = 
//        let isRemote, qName = parseQueueName queueName
//        if not isRemote then createQueueIfMissing qName
//        new MessageQueue(qName), isRemote
//
//    let SendMessageToQueue (messageQueue:MessageQueue) isTransactionalQueue (msg:Message) = 
//        isTransactionalQueue
//        |> function
//           | true -> 
//               use scope = new TransactionScope()
//               messageQueue.Send(msg, MessageQueueTransactionType.Automatic)
//               scope.Complete()
//           | _ -> messageQueue.Send(msg)        
//
//    let GetMessageBusImplDetails (queueName:string, isTransactionalQueue:bool) =
//        let messageQueue, isRemoteQueue = getMessageQueueInfo queueName
//        let isTransQueue = match isRemoteQueue with
//                           | true -> isTransactionalQueue
//                           | _ -> messageQueue.Transactional
//        { QueueName = queueName; IsTransactionalQueue = isTransQueue; MessageQueueInstance = messageQueue; IsRemoteQueue = isRemoteQueue
//          Publisher = SendMessageToQueue; IsForTesting = false }
//
//type MessageBus(implDetails) =    
//    let messageQueue = implDetails.MessageQueueInstance
//    new (queueName, isTransactionQueue) = new MessageBus( GetMessageBusImplDetails(queueName, isTransactionQueue) )
//    interface IMessageBus with
//        member x.QueueName with get () = implDetails.QueueName
//        
//        member x.Publish message = 
//            let msgTypeName = message.GetType().AssemblyQualifiedName
//            new Message(Label = msgTypeName, Body = message, Formatter = new BinaryMessageFormatter())
//            |> implDetails.Publisher messageQueue implDetails.IsTransactionalQueue  
//        
//        member x.Subscribe<'a> (success:Action<'a>) (failure:Action<Exception, obj>) =     
//            match implDetails.IsForTesting with
//            | true -> ()
//            | _ -> 
//                messageQueue.ReceiveCompleted.Add( 
//                    fun (args) -> 
//                        try                              
//                            args.Message.Formatter <- new BinaryMessageFormatter()
//                            args.Message.Body :?> 'a |> success.Invoke
//                        with
//                        | ex ->
//                            try
//                              failure.Invoke(ex, args.Message.Body)  
//                            with
//                            | ex -> raise ex
//                        messageQueue.BeginReceive() |> ignore)
//                messageQueue.BeginReceive() |> ignore
//
//        member x.Dispose () = messageQueue.Dispose()
//    member x.QueueName with get() = (x :> IMessageBus).QueueName
//    member x.Publish message = (x :> IMessageBus).Publish message    
//    member x.Subscribe<'a> (success:Action<'a>) (failure:Action<Exception, obj>) = (x :> IMessageBus).Subscribe<'a> success failure
//    member x.Dispose() = (x :> IMessageBus).Dispose()