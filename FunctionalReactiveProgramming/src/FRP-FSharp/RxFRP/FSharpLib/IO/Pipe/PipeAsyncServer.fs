module PipeAsyncServer

    open System
    open System.IO
    open System.IO.Pipes
    open System.Net
    open System.Text
    open System.ComponentModel
    open System.Threading

    type NamedPipeServerStream with
        member x.AsyncConnect() =
            Async.FromBeginEnd(x.BeginWaitForConnection, x.EndWaitForConnection)


//    type internal ServerPipeHanlder(pipeServer:NamedPipeServerStream, readCallback:(string -> unit)) =
        
        

    type ServerAsyncPipe(pipeName, readCallback:(string -> unit) option)=
        
        let serverPipe = new NamedPipeServerStream(pipeName, PipeDirection.InOut, -1, PipeTransmissionMode.Byte, PipeOptions.WriteThrough 
                                     ||| PipeOptions.Asynchronous, 0x100, 0x100) // Security ??
        do serverPipe.ReadMode <- PipeTransmissionMode.Byte
       
        let readCallback = defaultArg readCallback (fun s -> printfn "Message Received: %s" s)
        let log msg = printfn "%s" msg
        let token = new CancellationTokenSource()
        
        // if (serverPipe.CanRead)
        let startReadingStream f = 
            log (sprintf "Pipe Server is starting listening...")
            let rec loopReading bytes (sb:StringBuilder) = async {
                if serverPipe.IsConnected then 
                    let! bytesRead = serverPipe.AsyncRead(bytes,0,bytes.Length)
                    log (sprintf "Pipe Server readed %d bytes" bytesRead)
                    if bytesRead > 0 then
                        sb.Append(Encoding.Unicode.GetString(bytes, 0, bytesRead)) |> ignore
                        Array.Clear(bytes, 0, bytes.Length)
                        return! loopReading bytes sb
                    else
                        log (sprintf "Pipe Server message received and completed")
                        f (sb.ToString())
                        return! loopReading bytes (sb.Clear())
                else return () }
            Async.Start(loopReading (Array.zeroCreate<byte> 256) (new StringBuilder()), token.Token)

        member __.Write text =
            if serverPipe.IsConnected && serverPipe.CanWrite then
                log (sprintf "Pipe Server sending message %s" text)
                let write = async {
                    let message = Encoding.Unicode.GetBytes(text:string)
                    do! serverPipe.AsyncWrite(message,0, message.Length)
                    serverPipe.Flush() // Async 
                    serverPipe.WaitForPipeDrain() }
                Async.Start(write, token.Token) 


        member __.Connect() = 
            if not <| serverPipe.IsConnected then
                 let handleServer = async {
                    do! serverPipe.AsyncConnect()
                    log (sprintf "Pipe Server listening...")
                    startReadingStream readCallback
                    return () }
                 Async.Start(handleServer)
                     
//
//        let rec waitingForConnection = async {
//            let pipeServer = new NamedPipeServerStream(namePipeServer, PipeDirection.InOut, -1, PipeTransmissionMode.Message, PipeOptions.Asynchronous ||| PipeOptions.WriteThrough)
//            do! pipeServer.AsyncConnect()
//            Async.Start(handlePipeServer pipeServer)
//  
//            } 
           
      
            
            