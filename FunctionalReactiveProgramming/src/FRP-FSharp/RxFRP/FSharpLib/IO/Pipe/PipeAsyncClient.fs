module PipeAsyncClient
   
    open System
    open System.IO
    open System.IO.Pipes
    open System.Net
    open System.Text
    open System.ComponentModel
    open System.Threading

    type ClientAsyncPipe(namePipe:string, serverName:string option, readCallback:(string -> unit) option)=
        let serverName = defaultArg serverName "."
        let clientPipe = new NamedPipeClientStream(serverName, namePipe, PipeDirection.InOut, 
                                                               PipeOptions.Asynchronous ||| PipeOptions.WriteThrough,
                                                               Security.Principal.TokenImpersonationLevel.Impersonation)
        let log msg = printfn "%s" msg
        let token = new CancellationTokenSource()
        let readCallback = defaultArg readCallback (fun s -> printfn "Message Received: %s" s)
        
        let startReadingStream f = 
            log (sprintf "Pipe Client is starting listening...")
            let rec loopReading bytes (sb:StringBuilder) = async {
                let! bytesRead = clientPipe.AsyncRead(bytes,0,bytes.Length)
                log (sprintf "Pipe Client readed %d bytes" bytesRead)
                if bytesRead > 0 then
                    sb.Append(Encoding.Unicode.GetString(bytes, 0, bytesRead)) |> ignore
                    Array.Clear(bytes, 0, bytes.Length)
                    return! loopReading bytes sb
                else 
                    log (sprintf "Pipe Client message received and completed")
                    f (sb.ToString())
                    return! loopReading bytes (sb.Clear()) }
            Async.Start(loopReading (Array.zeroCreate<byte> 256) (new StringBuilder()), token.Token)
        
        member __.Write text = 
            if clientPipe.IsConnected && clientPipe.CanWrite then
                log (sprintf "Pipe Client sending message %s" text)
                let write = async {
                    let message = Encoding.Unicode.GetBytes(text:string)
                    do! clientPipe.AsyncWrite(message,0, message.Length)
                    clientPipe.Flush() // Async 
                    clientPipe.WaitForPipeDrain() }
                Async.Start(write, token.Token) 


        member __.Connect() =
            if not <| clientPipe.IsConnected then
                log (sprintf "Connecting Pipe Client...")
                clientPipe.Connect() 
                clientPipe.ReadMode <- PipeTransmissionMode.Byte
                log (sprintf "Pipe Client connected")
                startReadingStream readCallback
            { new IDisposable with
                member x.Dispose() =
                    token.Cancel()
                    clientPipe.Close()
                    clientPipe.Dispose() }
        
        member __.Stop =
            token.Cancel()
            clientPipe.Close()
        
       