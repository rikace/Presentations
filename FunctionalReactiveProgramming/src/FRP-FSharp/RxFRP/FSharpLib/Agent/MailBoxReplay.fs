namespace Easj360FSharp

module MailBoxReplay =

    let curry f x y = f (x, y)

    let (<->) (m:'a MailboxProcessor) msg = m.PostAndReply(fun replyChannel -> msg replyChannel)


    type Message = M of int 
                  | Fetch of int * AsyncReplyChannel<string> 
                  | Stop 
                  | Print of AsyncReplyChannel<string>
                   
    let a = 
            MailboxProcessor.Start(fun inbox ->
                let rec loop(i) = async {                
                    let! msg = inbox.Receive()
                    match msg with 
                    | M  x -> let result = x * x
                              printfn "Response %d"  result
                    | Stop -> ()
                    | Print inbox -> do inbox.Reply( i.ToString() )
                    | Fetch (x, inbox) -> do inbox.Reply( (x * x).ToString() )
                    return! loop(i + 1)
                }
                loop (0)
            )
        
    let r = Async.RunSynchronously( a.PostAndAsyncReply(fun x -> Fetch(4,x)) )
    //    a.Post(M(5))
    let re = a.PostAndReply(fun x -> Fetch(6,x))
    //    a.PostAndReply(fun x -> Print(x))

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
      
    //let buffer = new int BoundedBuffer(42)
    //buffer.Put 42
    //printfn "%d" (buffer.Get())
    //buffer.Stop()
