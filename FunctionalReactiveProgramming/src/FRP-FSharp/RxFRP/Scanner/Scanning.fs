// Adapted from: http://msdn.microsoft.com/en-us/library/ee370554.aspx
module Scanning
    open System
    open System.Threading

    let random = System.Random()
    type Result = double
    type Job = int * Async<Result> * CancellationTokenSource
    type Message = int * Result

    // Generates mock jobs by using Async.Sleep. 
    let createJob(id:int, source:CancellationTokenSource) =
        let job = async {
            // Let the time be a random number between 1 and 10000. 
            // The mock computed result is a floating point value. 
            let time = random.Next(10000)
            let result = random.NextDouble()
            let count = ref 1
            while (!count <= 1000 && not source.IsCancellationRequested) do 
                do! Async.Sleep(time / 500)
                count := !count + 1
            return result
            }
        id, job, source

    // This agent processes when jobs are completed. 
    let completeAgent = MailboxProcessor<Message>.Start(fun inbox ->
        let rec loop n =
            async {
                let! (id, result) = inbox.Receive()
                printfn "The result of job #%d is %f" id result
                do! loop <| n + 1
            }
        loop 0)

    // inprogressAgent maintains a queue of in-progress jobs that can be 
    // scanned to remove canceled jobs. It never runs its processor function, 
    // so we set it to do nothing. 
    let inprogressAgent = new MailboxProcessor<Job>(fun _ -> async { () })

    // This agent starts each job in the order in which it is received. 
    let runAgent = MailboxProcessor<Job>.Start(fun inbox ->
        let rec loop () =
            async {          
                let! (id, job, source) = inbox.Receive()
                printfn "Starting job #%d" id

                // Post to the in-progress queue.
                inprogressAgent.Post(id, job, source)
                // Start the job.
                Async.StartWithContinuations(job,
                    (fun result -> completeAgent.Post(id, result)),
                    (fun _ -> ()),
                    (fun cancelException -> printfn "Canceled job #%d" id),
                    source.Token)
                do! loop ()
                }
        loop ())

    let cancelJob(cancelId) =
        Async.RunSynchronously(
            inprogressAgent.Scan(fun (jobId, result, source) ->
                let action =
                    async {
                        printfn "Canceling job #%d" cancelId
                        source.Cancel()
                    }
                // Return Some(async) if the job ID matches. 
                if (jobId = cancelId) then
                    Some(action)
                else
                    None))

