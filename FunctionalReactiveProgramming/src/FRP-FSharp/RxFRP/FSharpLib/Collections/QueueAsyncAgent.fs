module QueueAsyncAgent

open System
open System.Threading
open System.Collections.Generic

type Agent<'T> = MailboxProcessor<'T>

////////////////////////// QueueAsyncAgent /////////////////////
type internal BlockingAgentAsyncMessage<'T> = 
  | Add of 'T * AsyncReplyChannel<unit> 
  | Get of AsyncReplyChannel<'T>

type QueueAsyncAgent<'T>() =
  [<VolatileField>]
  let mutable count = 0
  let agent = Agent.Start(fun agent ->    
    let queue = new Queue<_>()
    let rec emptyQueue() = 
      agent.Scan(fun msg ->
        match msg with 
        | Add(value, reply) -> Some(enqueueAndContinue(value, reply))
        | _ -> None )    
    and runningQueue() = async {
      let! msg = agent.Receive()
      match msg with 
      | Add(value, reply) -> return! enqueueAndContinue(value, reply)
      | Get(reply) -> return! dequeueAndContinue(reply) }
    and enqueueAndContinue (value, reply) = async {
      reply.Reply() 
      queue.Enqueue(value)
      count <- queue.Count
      return! chooseState() }
    and dequeueAndContinue (reply) = async {
      reply.Reply(queue.Dequeue())
      count <- queue.Count
      return! chooseState() }
    and chooseState() = 
      if queue.Count = 0 then emptyQueue()
      else runningQueue()
    emptyQueue() )

  member x.AsyncAdd(v:'T, ?timeout) = 
    agent.PostAndAsyncReply((fun ch -> Add(v, ch)), ?timeout=timeout)

  member x.AsyncGet(?timeout) = 
    agent.PostAndTryAsyncReply(Get, ?timeout=timeout)

  member x.Count = Thread.VolatileRead(ref count)

