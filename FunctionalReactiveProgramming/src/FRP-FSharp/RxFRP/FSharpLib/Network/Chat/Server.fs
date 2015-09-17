module Server

#light
open System
open System.IO
open System.Net
open System.Net.Sockets
open System.Text
open System.Threading
open System.Collections.Generic

// Enhance the TcpListener class so it can handle async connections
type System.Net.Sockets.TcpListener with
    member x.AsyncAcceptTcpClient() =
        Async.FromBeginEnd(x.BeginAcceptTcpClient, x.EndAcceptTcpClient)

// Type that defines protocol for interacting with the ClientTable
type ClientTableCommands = 
    | Add of (string * StreamWriter)
    | Remove of string
    | SendMessage of string
    | ClientExists of (string * AsyncReplyChannel<bool>)

// A class that will store a list of names of connected clients along with 
// streams that allow the client to be written too
type ClientTable() =
    // create the mail box 
    let mailbox = MailboxProcessor.Start(fun inbox ->
        // main loop that will read messages and update the
        // client name/stream writer map
        let rec loop (nameMap: Map<string, StreamWriter>) =
            async { let! msg = inbox.Receive()
                    match msg with
                    | Add (name, sw) ->
                        return! loop (Map.add name sw nameMap)
                    | Remove name -> 
                        return! loop (Map.remove name nameMap)
                    | ClientExists (name, rc) -> 
                        rc.Reply (nameMap.ContainsKey name) 
                        return! loop nameMap
                    | SendMessage msg -> 
                        for (_, sw) in Map.toSeq nameMap do
                            try
                                sw.WriteLine msg
                                sw.Flush()
                            with _ -> ()
                        return! loop nameMap }
        // start the main loop with an empty map
        loop Map.empty)
    /// add a new client
    member x.Add(name, sw) = mailbox.Post(Add(name, sw))
    /// remove an existing connection
    member x.Remove(name) = mailbox.Post(Remove name)
    /// handles the process of sending a message to all clients
    member x.SendMessage(msg) = mailbox.Post(SendMessage msg)
    /// checks if a client name is taken
    member x.ClientExists(name) = mailbox.PostAndReply(fun rc -> ClientExists(name, rc))

/// perform async read on a network stream passing a continuation 
/// function to handle the result 
let rec asyncReadTextAndCont (stream: NetworkStream) cont  =
    // unfortunatly we need to specific a number of bytes to read
    // this leads to any messages longer than 512 being broken into
    // different messages
    async { let buffer = Array.create 512 0uy
            let! read = stream.AsyncRead(buffer, 0, 512) 
            let allText = Encoding.UTF8.GetString(buffer, 0, read)
            return cont stream allText  }

// class that will handle client connections
type Server() =

    // client table to hold all incoming client details
    let clients = new ClientTable()
    
    // handles each client
    let handleClient (connection: TcpClient) =
        // get the stream used to read and write from the client
        let stream = connection.GetStream()
        // create a stream write to more easily write to the client
        let sw = new StreamWriter(stream)
        
        // handles reading the name then starts the main loop that handles
        // conversations
        let rec requestAndReadName (stream: NetworkStream) (name: string) =
            // read the name
            let name = name.Replace(Environment.NewLine, "")
            // main loop that handles conversations
            let rec mainLoop (stream: NetworkStream) (msg: string) = 
                try
                   // send received message to all clients
                    let msg = Printf.sprintf "%s: %s" name msg
                    clients.SendMessage msg
                with _ ->
                    // any error reading a message causes client to disconnect
                    clients.Remove name
                    sw.Close()
                Async.Start (asyncReadTextAndCont stream mainLoop)
            if clients.ClientExists(name) then
                // if name exists print error and relaunch request
                sw.WriteLine("ERROR - Name in use already!")
                sw.Flush() 
                Async.Start (asyncReadTextAndCont stream requestAndReadName)
            else 
                // name is good lanch the main loop
                clients.Add(name, sw)
                Async.Start (asyncReadTextAndCont stream mainLoop)
        // welcome the new client by printing "What is you name?"
        sw.WriteLine("What is your name? "); 
        sw.Flush()
        // start the main loop that handles reading from the client
        Async.Start (asyncReadTextAndCont stream requestAndReadName)
            
    // create a tcp listener to handle incoming requests
    let listener = new TcpListener(IPAddress.Loopback, 4242)

    // main loop that handles all new connections
    let rec handleConnections() =
        // start the listerner
        listener.Start()
        if listener.Pending() then
            // if there are pending connections, handle them
            async { let! connection = listener.AsyncAcceptTcpClient()
                    printfn "New Connection"
                    // use a thread pool thread to handle the new request 
                    ThreadPool.QueueUserWorkItem(fun _ -> handleClient connection) |> ignore
                    // loop 
                    return! handleConnections() }
        else
            // no pending connections, just loop
            Thread.Sleep(1)
            async { return! handleConnections() }
            
    /// allow tot
    member server.Start() = Async.RunSynchronously (handleConnections())
// start the server class
(new Server()).Start()
