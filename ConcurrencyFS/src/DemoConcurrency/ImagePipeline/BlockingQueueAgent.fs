module ImagePipeline.BlockingQgent

open System
open System.IO
open System.Threading
open System.Collections.Generic

type Agent<'T> = MailboxProcessor<'T>


type internal BlockingAgentMessage<'T> = 
  | Add of 'T * AsyncReplyChannel<unit> 
  | Get of AsyncReplyChannel<'T>


type BlockingQueueAgent<'T>(maxLength) =
  [<VolatileField>]
  let mutable count = 0

  let agent = Agent.Start(fun agent ->
    let queue = new Queue<_>()
    let pending = new Queue<_>()

    let rec emptyQueue() = 
      agent.Scan(fun msg ->
        match msg with 
        | Add(value, reply) -> Some <| async {  queue.Enqueue(value)
                                                count <- queue.Count
                                                reply.Reply()
                                                return! nonEmptyQueue() }
        | _ -> None )

    and nonEmptyQueue() = async {
      let! msg = agent.Receive()
      match msg with 
      | Add(value, reply) -> 
          if queue.Count < maxLength then 
            queue.Enqueue(value)
            count <- queue.Count
            reply.Reply()
          else 
            pending.Enqueue(value, reply) 
          return! nonEmptyQueue()
      | Get(reply) -> 
          let item = queue.Dequeue()
          while queue.Count < maxLength && pending.Count > 0 do
            let itm, caller = pending.Dequeue()
            queue.Enqueue(itm)
            caller.Reply()
          count <- queue.Count
          reply.Reply(item)
          if queue.Count = 0 then return! emptyQueue()
          else return! nonEmptyQueue() }
    emptyQueue() )


  member x.Count = count

  member x.AsyncAdd(v:'T, ?timeout) = 
    agent.PostAndAsyncReply((fun ch -> Add(v, ch)), ?timeout=timeout)

  member x.AsyncGet(?timeout) = 
    agent.PostAndAsyncReply(Get, ?timeout=timeout)
