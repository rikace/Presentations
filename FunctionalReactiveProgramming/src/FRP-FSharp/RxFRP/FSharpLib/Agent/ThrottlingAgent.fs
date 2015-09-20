namespace Easj360FSharp

open System

module ThrottlingAgent =


    /// Message type used by the agent - contains queueing 
    /// of work items and notification of completion 
    type internal ThrottlingAgentMessage = 
      | Completed
      | Work of Async<unit>
      | WorkReply of int * AsyncReplyChannel<int>
    
    /// Represents an agent that runs operations in concurrently. When the number
    /// of concurrent operations exceeds 'limit', they are queued and processed later
    type ThrottlingAgent(limit) = 
      let agent = MailboxProcessor.Start(fun agent -> 

        /// Represents a state when the agent is blocked
        let rec waiting () = 
          // Use 'Scan' to wait for completion of some work
          agent.Scan(function
            | Completed -> Some(working (limit - 1))
            | _ -> None)

        /// Represents a state when the agent is working
        and working count = async { 
            // Receive any message 
            let! msg = agent.Receive()
            match msg with 
            | Completed -> 
                // Decrement the counter of work items
                return! working (count - 1)
            ////
            | WorkReply (n, reply) -> 
                async {
                    let res = n * n
                    do! Async.Sleep(1000)
                    printfn "Computing"
                    reply.Reply(res)
                    agent.Post(Completed) }
                |> Async.Start
                if count < limit then return! working (count + 1)
                else return! waiting ()
            ////
            | Work work ->
                // Start the work item & continue in blocked/working state
                async { try do! work 
                        finally agent.Post(Completed) }
                |> Async.Start
                if count < limit then return! working (count + 1)
                else return! waiting () }

        // Start in working state with zero running work items
        working 0)      

      /// Queue the specified asynchronous workflow for processing
      member x.DoWork(work) = agent.Post(Work work)
      member x.DoWorkReply(n) = agent.PostAndAsyncReply((fun reply -> WorkReply(n, reply)))

//let ag = ThrottlingAgent(1)
//let res = ag.DoWorkReply(9) 
//res |> Async.RunSynchronously
//
//Async.StartWithContinuations( 
//                        ag.DoWorkReply(7) ,
//                        (fun cont -> printfn "result"), // %d" cont),
//                        (fun exn -> printfn "Error"),
//                        (fun cancel -> printfn "cancel"))
//
//
// let superAwesomeAsyncTask = async { return 42 }
// let res89 = ag.DoWorkReply(89)
//
// Async.StartWithContinuations(
//        ag.DoWorkReply(89),
//        (fun (result : int) -> printfn "Task was completed with result %d" result),
//        (fun (exn) -> printfn "threw exception"),
//        (fun (oce) -> printfn "OCE"))
