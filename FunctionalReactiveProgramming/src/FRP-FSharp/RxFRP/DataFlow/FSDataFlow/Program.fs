open System
open System.Collections.Generic
open System.Reactive

type Agent<'T> = MailboxProcessor<'T>

module ThrottlingAgnet =

    /// Message type used by the agent - contains queueing 
    /// of work items and notification of completion 
    type internal ThrottlingAgentMessage<'a, 'b>= 
      | Completed of 'b
      | Get of AsyncReplyChannel<'b>
      | Work of 'a
    
    /// Represents an agent that runs operations in concurrently. When the number
    /// of concurrent operations exceeds 'limit', they are queued and processed later
    type ThrottlingAgent<'a, 'b>(f:'a -> 'b, limit) = 
      let agent = MailboxProcessor.Start(fun agent -> 

        let queue = Queue<_>()
        /// Represents a state when the agent is blocked
        let rec waiting () = 
          // Use 'Scan' to wait for completion of some work
          agent.Scan(function
            | Completed(b) ->   queue.Enqueue b
                                Some(working (limit - 1))
            | _ -> None)

        /// Represents a state when the agent is working
        and working count = async { 
          // Receive any message 
          let! msg = agent.Receive()
          match msg with 
          | Completed(b) ->
              queue.Enqueue b 
              // Decrement the counter of work items
              return! working (count - 1)
          | Work work ->
              // Start the work item & continue in blocked/working state
              async {   let res = f work // TODO CATCH ERROR
                        agent.Post(Completed(res)) }
              |> Async.Start
              if count < limit - 1 then return! working (count + 1)
              else return! waiting () }

        // Start in working state with zero running work items
        working 0)      

      /// Queue the specified asynchronous workflow for processing
      member x.DoWork(work) = agent.Post(Work work)

//module Throttling =
//    // We can ask the agent to enqueue a new work item;
//    // and the agent sends itself a completed notification
//    type ThrottlingMessage<'a, 'b> = 
//      | Enqueue of 'a
//      | Completed of 'b
//      | Dequeue of AsyncReplyChannel<'b option>
// 
//
//    type ThrottlingAgent<'a, 'b>(f:'a -> 'b, ?limit) = 
//    
//        let limit = defaultArg limit 1
//
//        let agent = MailboxProcessor.Start(fun inbox -> async {
//              // The agent body is not executing in parallel, 
//              // so we can safely use mutable queue & counter 
//              let queue = System.Collections.Generic.Queue<_>()
//              let outqueue = System.Collections.Generic.Queue<_>()
//              let rec loop running = async {
//                let! msg = inbox.Receive()
//                match msg with
//                | Dequeue(reply) -> let item = outqueue.Dequeue()
//                                    reply.Reply(item)
//                | Completed(r) -> outqueue.Enqueue r
//                                  decr running
//                | Enqueue w ->  queue.Enqueue(w)
//                            
//                while running.Value < limit && queue.Count > 0 do
//                  let work = queue.Dequeue()
//                  incr running
//                  do! 
//                    // When the work completes, send 'Completed'
//                    // back to the agent to free a slot
//                    async { let res = f work
//                            inbox.Post(Completed(res)) } 
//                    |> Async.StartChild
//                    |> Async.Ignore  } 
//              loop (ref 0)} )
//
//
//        member x.Post(msg:'a) =
//            agent.Post(Enqueue msg)

module QueueAgent =

    type internal BlockingAgentMessage<'a,'b> = 
      | Add of 'a * AsyncReplyChannel<unit> 
      | Get of AsyncReplyChannel<'b>
      | Complete of 'b


    // No work to do 
    // full work - get or complete
    // runing - add - get - complete
    type BlockingQueueAgent<'a, 'b>(f:'a -> 'b, maxLength) =
      let running = ref 0

      let agent = Agent.Start(fun agent ->
    
        let inputItem = new Queue<'a>()
        let outputItem = new Queue<'b>()

        let rec emptyQueue() = 
            printfn "EmptyQueue"
            agent.Scan(fun msg ->
                match msg with
                | Add(value, reply) -> //incr running
                                       Some(enqueueAndContinue(value, reply))
                | Complete(b) -> decr running
                                 outputItem.Enqueue b
                                 Some(chooseState())
                | _ -> None )
        and fullQueue() =
          printfn "FullQueue" 
          agent.Scan(fun msg ->
            match msg with 
            | Get(reply) -> Some(dequeueAndContinue(reply))
            | Complete(b) -> decr running
                             outputItem.Enqueue b
                             Some(runningQueue())
            | _ -> None )
        and runningQueue() = async {
          printfn "runningQueue"
          let! msg = agent.Receive()
          match msg with 
          | Add(value, reply) -> return! enqueueAndContinue(value, reply)
          | Complete(b) ->  decr running
                            outputItem.Enqueue b
                            return! runningQueue()
          | Get(reply) -> return! dequeueAndContinue(reply) }

        and enqueueAndContinue (value, reply) = async {
          reply.Reply() 
          inputItem.Enqueue value 
         // incr running 
          return! chooseState() }

        and dequeueAndContinue (reply) = async {
          reply.Reply(outputItem.Dequeue())
          return! chooseState() }
        
        and chooseState() = 
            while running.Value < maxLength && inputItem.Count > 0 do
                let item = inputItem.Dequeue()
                incr running
                async { let res = f item
                        agent.Post(Complete(res)) } |> Async.Start
            if !running = 0 && outputItem.Count > 0 then runningQueue()
            elif !running = 0 then emptyQueue()
            elif !running < maxLength then runningQueue()
           // elif !runningQueue = maxLength then fullQueue()
            else fullQueue()
        // Start with an empty queue
        emptyQueue() )

      /// Asynchronously adds item to the queue. The operation ends when
      /// there is a place for the item. If the queue is full, the operation
      /// will block until some items are removed.
      member x.AsyncAdd(v:'T, ?timeout) = 
        agent.PostAndAsyncReply((fun ch -> Add(v, ch)), ?timeout=timeout)

      /// Asynchronously gets item from the queue. If there are no items
      /// in the queue, the operation will block unitl items are added.
      member x.AsyncGet(?timeout) = 
        agent.PostAndAsyncReply(Get, ?timeout=timeout)

//      /// Gets the number of elements currently waiting in the queue.
//      member x.Count = count

module AgentStuff =
    
    [<InterfaceAttribute>]
    type IAgentPost<'a, 'b> =
        abstract PostAction : msg:'a -> 'b

//    type AgentAction<'a>(f: 'a -> unit) =
//        
//        let agentAction  = 
//                 Agent<_>.Start(fun inbox ->
//                        let rec loop n = async {
//                            let! msg = inbox.Receive()
//                            f msg
//                            return! loop (n + 1)
//                                }
//                        loop 0 )
//           
//
//        member x.PostAction(msg:'a) =
//                agentAction.Post(msg)
//
//        member x.SendAction(msg:'a) =
//                agentAction.Post(msg)



    type AgentTransformer<'a,'b>(f:'a -> 'b) =
        
            let subject = new System.Reactive.Subjects.Subject<'b>()
        
            let agent = Agent<_>.Start(fun inbox ->
                    let rec loop observers = async {
                        let! msg = inbox.Receive()
                        match msg with
                        | ObserverAdd o -> return! loop (o::observers)
                        | ObserverRemove o -> return! loop (observers |> List.filter(fun f -> f <> o))
                        | PostAction(a) -> 
                                let res:'b = f a
                                observers |> List.iter(fun o -> o.OnNext(res))
                                return! loop observers
                        | PostFunc(a,reply) -> 
                                let res:'b = f a
                                reply.Reply(res)
                                observers |> List.iter(fun o -> o.OnNext(res))
                                return! loop observers }
                    loop [] )

            abstract member Post : 'a -> unit
            default this.Post (msg:'a) = 
                agent.Post(PostAction msg)
                
            abstract member PostAndReply : 'a -> Async<'b>
            default this.PostAndReply (msg:'a) = 
                agent.PostAndAsyncReply(fun (reply:AsyncReplyChannel<'b>) -> PostFunc(msg, reply))

            interface IDisposable with
                member __.Dispose() =
                        (agent :> IDisposable).Dispose() 

            interface IObservable<'b> with
                member x.Subscribe(observer:IObserver<'b>) =
                    observer |> ObserverAdd |> agent.Post
                    { new IDisposable with
                        member x.Dispose() =
                            observer |> ObserverRemove |> agent.Post }

    and AgentAction<'a>(f:'a -> unit) =
        inherit AgentTransformer<'a, unit>(f)
                    
    and private Message<'a, 'b> =
        | ObserverRemove of IObserver<'b>
        | ObserverAdd of IObserver<'b>
        | PostFunc of 'a * AsyncReplyChannel<'b>
        | PostAction of 'a  

//    type AgentTransformer<'a,'b> with
//        member x.LinkTo(agentLink:AgentTransformer<'b, 'c>) = async {
//          //  x.Subscribe(agentLink)
//        }
//        
    type TFUn<'a,'b> =
        | F1 of ('a -> unit)
        | F2 of ('a -> 'b)

open AgentStuff
open QueueAgent

[<EntryPoint>]
let main argv = 


    let q = QueueAgent.BlockingQueueAgent<int, string>((fun x -> printfn "%d" (x * x)   
                                                                 System.Threading.Thread.Sleep(60000)
                                                                 string (x * x)), 2)
    q.AsyncAdd(6) |> Async.Start
    q.AsyncGet() |> Async.RunSynchronously |> (printfn "%s")
//    // To use the throttling agent, call it with a specified limit
//    // and then add items using the 'Enqueue' message!
//    let w = throttlingAgent 5 
//    for i in 0 .. 20 do 
//      async { printfn "Starting %d - Thread id %d" i System.Threading.Thread.CurrentThread.ManagedThreadId
//              do! Async.Sleep(1000)
//              printfn "Done %d" i  }
//      |> Enqueue
//      |> w.Post


    let rec fib = function
      | 0 | 1 -> 1
      | n -> fib (n-1) + fib (n-2)

    let rec fib_cps n k = 
      match n with
      | 0 | 1 -> k 1
      | n -> fib_cps (n-1) (fun a -> fib_cps (n-2) (fun b -> k (a+b)))

    fib_cps 10 (fun b -> b)

    fib 10



    let rec fact n k =
        if n = 0 then
            k(1)
        else fact (n - 1) (fun x -> //printfn "%A" n
                                    k(n * x))



    let map f list cont = 
      let rec loop acc list cont = 
        match list with
        | [] -> cont (List.rev acc) // Reverse the list at the end & call continuation!
        | x::xs -> f x (fun x' ->   // Call `f` with `cont` that recursively calls `loop`
            loop (x'::acc) xs cont )// Call `loop` with newly projected element in `acc`
      loop [] list cont


    map (fun n cont -> cont (n * 2)) [1 .. 10] (printfn "%A")


    fact 5 (printfn "%d")

    let obs = Observable.Subject.t<string>()



    
    let a = new AgentAction<string>(fun s -> printfn "%s" s)
    a.Post("Ciao")

    let t = AgentTransformer<int, string>(fun s -> let r = s * s 
                                                   printfn "result %s" (string r)
                                                   string r)


    let t' = AgentTransformer<int,unit>(fun s -> let r = s * s 
                                                 printfn "result %s" (string r))




    t.Subscribe(fun s -> printfn "Hello from Sub %s" s) |> ignore
                     
    //t.LinkTo()
                                                         
    let res = t.Post(7)

    printf "%A" res

    Console.ReadKey() |> ignore
    0 // return an integer exit code

