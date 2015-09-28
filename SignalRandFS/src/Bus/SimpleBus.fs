namespace Bus

module SimpleBus =

    open System
    open System.Messaging
    open System.Threading

    let private createQueueIfMissing (queueName:string) =
        if not (MessageQueue.Exists queueName) then
            MessageQueue.Create queueName |> ignore

    let private parseQueueName (queueName:string) =             
        let fullName = match queueName.Contains("@") with
                       | true when queueName.Split('@').[1] <> "localhost" -> 
                           queueName.Split('@').[1] + "\\private$\\" + 
                               queueName.Split('@').[0]
                       | _ -> ".\\private$\\" + queueName
        createQueueIfMissing fullName
        fullName

    let subscribe<'a> queueName success failure =     
        let queue = new MessageQueue(parseQueueName queueName)

        queue.ReceiveCompleted.Add( 
            fun (args) ->     
                try        
                    args.Message.Formatter <- new BinaryMessageFormatter() 
                    args.Message.Body :?> 'a |> success
                with
                | ex -> failure ex args.Message.Body
                queue.BeginReceive() |> ignore)

        queue.BeginReceive() |> ignore
        queue

    let send message =
        ()

    let publish queueName  message =     
        use queue = new MessageQueue(parseQueueName queueName)
        new Message(message, new BinaryMessageFormatter())
        |> queue.Send

module PublishMonad =
    open System
    open System.Messaging
    open System.Threading

 
    type SendCommandWith<'a> = SendCommandWith of 'a

    type RetryPublishBuilder(f, max, sleepMilliseconds : int) = 
        member x.Return(a) = a
        member x.Bind(SendCommandWith(msg):SendCommandWith<_>, fn) =
                let rec loop(n) = async {
                    if n = 0 then failwith "Failed"
                    else 
                        try 
                            f msg
                        with ex -> 
                            sprintf "Call failed with %s. Retrying." ex.Message |> printfn "%s"
                            do! Async.Sleep sleepMilliseconds
                            return! loop(n-1) }
                loop max |> Async.Start


//    let post n = agentS.Post n
//
//    let retry = RetryPublishBuilder(agentS.Post, 3, 500)
//
//    retry {
//           do! SendCommandWith("sample_queue")
//    }
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
