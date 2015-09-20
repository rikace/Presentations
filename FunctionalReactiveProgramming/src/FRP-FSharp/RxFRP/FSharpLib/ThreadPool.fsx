module ThreadPool 
open System
open System.Collections.Generic

type Agent<'T> = MailboxProcessor<'T>

type internal PoolAgentMessage = 
  | Completed 
  | AddJob of Async<unit>
  | GetJob of AsyncReplyChannel<Async<unit>>
  | GetCount of AsyncReplyChannel<int>

type AgentManagerMessage<'T> =
  | Add of Async<'T> 

  // use agent.Scan ? waiting for complete is best practice ?
  // pub sub patterm ?? agent as dispatcher
  // send-replay message to agent that is Async ?
  // process async method and then return value ?? Event or async Replay
type internal PoolAgent(coordinator:AgentCoordinator) as this =
    [<VolatileField>]
    let mutable count = 0
    let agent = Agent<PoolAgentMessage>.Start(fun agent -> 
            let doJob job = async { try do! job 
                                    finally agent.Post(Completed) }|> Async.Start

            let rec loop (queue:LinkedList<_>) count = async {
                let! msg = agent.TryReceive(1000)
                match msg with
                | None -> coordinator.GetJob(this)
                          return! loop queue count
                | Some(action) -> 
                        match action with
                        | Completed ->  if count > 0 then
                                            let job = queue.First
                                            queue.RemoveFirst()
                                            doJob job.Value
                                        return! loop queue (count - 1)
                        | AddJob job -> if count > 0 then 
                                            queue.AddLast(LinkedListNode(job))
                                        else
                                            doJob job
                                        return! loop queue (count + 1)
                        | GetJob(replay) -> let job = queue.Last
                                            queue.RemoveLast()
                                            replay.Reply(job.Value)
                                            return! loop queue (count - 1)
                        | GetCount(replay) -> replay.Reply(count)
                                              return! loop queue count }
            loop (new LinkedList<_>()) 0 )

    member x.Add(action:Async<unit>) = agent.Post(AddJob action)
    member x.Get() = agent.PostAndAsyncReply(fun replay -> GetJob(replay))
    member x.Count = agent.PostAndAsyncReply(fun replay -> GetCount(replay))

and AgentCoordinator(count) as this =
        let workers = Array.init count (fun i -> PoolAgent(this))
        let agent = MailboxProcessor<Async<unit>>.Start(fun inbox ->
                            let rec loop i = async {
                                let! msg = inbox.Receive()
                                let worker = workers 
                                              |> Array.minBy(fun f -> 
                                                    // read directly volatile filed is cheating by Tomas says that it's ok, other option shoud be to use Async.Parallel
                                                    Async.StartWithContinuations(f.Count))
                                worker.Add(msg)
                                return! loop (i+1) }
                            loop 0 )
        member x.AddJob(action:Async<unit>) = agent.Post(action)
        member internal x.GetJob(poolAgent:PoolAgent) = let workerMax = workers |> Array.maxBy(fun f -> Async.RunSynchronously(f.Count))
                                                        async{  let! job = workerMax.Get() // TODO Option return 
                                                                poolAgent.Add(job) } |> Async.Start