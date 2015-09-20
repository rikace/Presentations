//25-30
module Agents

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
    