module PipeClient
    open System
    open System.IO
    open System.IO.Pipes
    open System.Net
    open System.ComponentModel
    open System.Threading

    type ClientSync() = 

            static member private readPipeStream (pipeClient:NamedPipeClientStream) =
                use reader = new StreamReader(pipeClient)
                let rec readStream state =
                    let message = reader.ReadLine()
                    printfn "Received from server: %s" message
                    readStream state
                readStream []

            static member StartClientPipe(namedPipe:string) =
                use pipeClient = new NamedPipeClientStream(".", namedPipe, PipeDirection.In)
                printfn "Attempting to connect to pipe..."
                pipeClient.Connect()

                printfn "Connected to pipe."
                printfn "There are currently %d pipe server instances open." pipeClient.NumberOfServerInstances

                ClientSync.readPipeStream pipeClient