//25-30
module Agents =

    type Agent<'T> = MailboxProcessor<'T>

// one agent
 
    let oneAgent =
       Agent.Start(fun inbox ->
         async { while true do
                   let! msg = inbox.Receive()
                   printfn "got message '%s'" msg } )
 
    oneAgent.Post "hi"

// 100k agents
    let alloftheagents =
        [ for i in 0 .. 100000 ->
           Agent.Start(fun inbox ->
             async { while true do
                       let! msg = inbox.Receive()
                       if i % 10000 = 0 then
                           printfn "agent %d got message '%s'" i msg })]
 
    for agent in alloftheagents do
        agent.Post "ping!"

// error handling
    let errorAgent =
       Agent<int * System.Exception>.Start(fun inbox ->
         async { while true do
                   let! (agentId, err) = inbox.Receive()
                   printfn "an error '%s' occurred in agent %d" err.Message agentId })
 
    let agents =
       [ for agentId in 0 .. 10000 ->
            let agent =
                new Agent<string>(fun inbox ->
                   async { while true do
                             let! msg = inbox.Receive()
                             if msg.Contains("agent 99") then
                                 failwith "fail!" })
            agent.Error.Add(fun error -> errorAgent.Post (agentId,error))
            agent.Start()
            (agentId, agent) ]
 
    for (agentId, agent) in agents do
       agent.Post (sprintf "message to agent %d" agentId )
    

// 40-45
// Fail if agent ID contains "99". 
module Supervisors = 
    type Agent<'T> = MailboxProcessor<'T>
    // error handling
    let errorAgent =
        Agent<int * System.Exception>.Start(fun inbox ->
            async { while true do
                    let! (agentId, err) = inbox.Receive()
                    printfn "an error '%s' occurred in agent %d" err.Message agentId })
 
    let agents =
        [ for agentId in 0 .. 10000 ->
            let agent =
                new Agent<string>(fun inbox ->
                    async { while true do
                                let! msg = inbox.Receive()
                                if msg.Contains("agent 99") then
                                    failwith "fail!" })
            agent.Error.Add(fun error -> errorAgent.Post (agentId,error))
            agent.Start()
            (agentId, agent) ]
 
    for (agentId, agent) in agents do
        agent.Post (sprintf "message to agent %d" agentId )



// Using Async.Catch
// Reading, writing to file simultaneously will fail. 
module AsyncCatch = 
    open System
    open System.IO

    let writeToFile filename numBytes = 
        async {
            use file = File.Create(filename)
            printfn "Writing to file %s." filename
            do! file.AsyncWrite(Array.zeroCreate<byte> numBytes)
        }

    let readFile filename numBytes =
        async {
            use file = File.OpenRead(filename)
            printfn "Reading from file %s." filename
            do! file.AsyncRead(numBytes) |> Async.Ignore
        }

    let filename = "BigFile.dat" 
    let numBytes = 100000000

    let result1 = writeToFile filename numBytes
                 |> Async.Catch
                 |> Async.RunSynchronously
    match result1 with
    | Choice1Of2 _ -> printfn "Successfully wrote to file."; ()
    | Choice2Of2 exn -> 
          printfn "Exception occurred writing to file %s: %s" filename exn.Message

    // Start these next two operations asynchronously, forcing an exception due 
    // to trying to access the file twice simultaneously.
    Async.Start(readFile filename numBytes)
    let result2 = writeToFile filename numBytes
                 |> Async.Catch
                 |> Async.RunSynchronously
    match result2 with
    | Choice1Of2 buffer -> printfn "Successfully read from file."
    | Choice2Of2 exn ->
        printfn "Exception occurred reading from file %s: %s" filename (exn.Message)

// Also see Emailer for more supervisor handling.
// Also see Replier for Async.StartWithContinuations. * 


// 45-50
// Same as async example. Computes length of info on homepage. 
module ScalingAgents =

    open System.Net

    type Agent<'T> = MailboxProcessor<'T>

    let urlList = [ ("Microsoft.com", "http://www.microsoft.com/");
                    ("MSDN", "http://msdn.microsoft.com/");
                    ("Google", "http://www.google.com") ]

    let processingAgent() = Agent<string * string>.Start(fun inbox ->
                            async { while true do
                                    let! name,url = inbox.Receive()
                                    let uri = new System.Uri(url)
                                    let webClient = new WebClient()
                                    let! html = webClient.AsyncDownloadString(uri)
                                    printfn "Read %d characters for %s" html.Length name })

    let scalingAgent : Agent<(string * string) list> = Agent.Start(fun inbox -> 
                                        async { while true do 
                                                let! msg = inbox.Receive()
                                                msg
                                                |> List.iter (fun x -> 
                                                                let newAgent = processingAgent()
                                                                newAgent.Post x )})

    scalingAgent.Post urlList
