
module BoundedBuffer 
    let curry f x y = f (x, y)

    let (<->) (m:'a MailboxProcessor) msg = m.PostAndReply(fun replyChannel -> msg replyChannel)

    type 'a BufferMessage = Put of 'a * unit AsyncReplyChannel 
                          | Get of 'a AsyncReplyChannel 
                          | Stop of unit AsyncReplyChannel

    type 'a BoundedBuffer(N:int) =  
      let buffer =     
        MailboxProcessor.Start(fun inbox ->
          let buf:'a array = Array.zeroCreate N
          let rec loop in' out n =
            async { let! msg = inbox.Receive()
                    match msg with
                    | Put (x, replyChannel) when n < N ->
                        Array.set buf in' x
                        replyChannel.Reply ()
                        return! loop ((in' + 1) % N) out (n + 1)

                    | Get replyChannel when n > 0 ->
                        let r = Array.get buf out
                        replyChannel.Reply r
                        return! loop in' ((out + 1) % N) (n - 1)

                    | Stop replyChannel -> replyChannel.Reply(); return () }
          loop 0 0 0)
      
      member this.Put(x:'a) = buffer <-> curry Put x
      member this.Get() = buffer <-> Get
      member this.Stop() = buffer <-> Stop
      
//let buffer = new int BoundedBuffer.BoundedBuffer(42)
//buffer.Put 42
//printfn "%d" (buffer.Get())
//buffer.Stop()


// let buffer = new int BoundedBuffer 42;;
//
//val buffer : int BoundedBuffer
//
//> buffer.Put 12;;
//val it : unit = ()
//> buffer.Put 34;;
//val it : unit = ()
//> buffer.Put 56;;
//val it : unit = ()
//> buffer.Get();;
//val it : int = 12
//> buffer.Get();;
//val it : int = 34
//> buffer.Get();;
//val it : int = 56
//> buffer.Stop();;
//val it : unit = ()