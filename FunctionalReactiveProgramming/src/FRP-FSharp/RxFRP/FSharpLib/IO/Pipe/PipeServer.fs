module PipeServer 
    open System
    open System.IO
    open System.IO.Pipes
    open System.Net
    open System.ComponentModel
    open System.Threading

    type ServerSync() = 
            static member private writePipeServer (pipeServer:NamedPipeServerStream) =
                use sw = new StreamWriter(pipeServer)
                sw.AutoFlush <- true
                let rec loop state = 
                    printf "Enter Text: "
                    let text = System.Console.ReadLine()
                    printfn "%s" text
                    if not <| String.IsNullOrWhiteSpace text then 
                        sw.WriteLine(text) // use async
                        loop state
                    else ()
                loop []

            static member StartServer(namePipe:string) = 
                use pipeServer = new NamedPipeServerStream(namePipe, PipeDirection.Out, 2)
                printfn "NamedPipeServerStream object created."
                printfn "Waiting for client connection..."
                pipeServer.WaitForConnection()

                printfn "Client connected"

                try
                    ServerSync.writePipeServer pipeServer
                with
                :? IOException as ioEx -> printfn "IO Error Message: %s" ioEx.Message
                | ex -> printfn "Error Message: %s" ex.Message


            
        

(*

“public static async Task Go() {
   // Start the server, which returns immediately because
   // it asynchronously waits for client requests
   StartServer()f; // This returns void, so compiler warning to deal with”

“   // Make lots of async client requestsf; save each client's Task<String>
   List<Task<String>> requests = new List<Task<String>>(10000)f;
   for (Int32 n = 0f; n < requests.Capacityf; n++)
      requests.Add(IssueClientRequestAsync("localhost", "Request #" + n))f;

   // Asynchronously wait until all client requests have completed
   // NOTE: If 1+ tasks throws, WhenAll rethrows the last-throw exception
   String[] responses = await Task.WhenAll(requests)f;

   // Process all the responses
   for (Int32 n = 0f; n < responses.Lengthf; n++)
      Console.WriteLine(responses[n])f;
}”*)