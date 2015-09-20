namespace Easj360FSharp

module JobAgentsWithCancellation =
    
    open System 
    open System.Threading 

    let random = System.Random() 

    // The state of the job.    
    type JobState =             
        | NotStarted            
        | Running               
        | Canceled              
        | Completed             
                                
    type Result = double        
                                
    // a Job consists of a job ID, a computation, a cancellation token source,
    // and a state.                                                           
    type Job = {                                                              
        id : int                                                              
        comp : Async<Result>                                                  
        token : CancellationToken
        state : JobState
        }               
                        
    type Message = int * Result
 
    // Generates mock jobs using Async.Sleep
    let createJob(id:int, token:CancellationToken) =                            
        let job = async {                                                       
            // Let the time be a random number between 1 and 10000              
            // And the mock computed result is a floating point value           
            let time = random.Next(10000)                                       
            let result = 2.0 + random.NextDouble()                              
            let count = ref 0                                                   
            // Start child jobs                                                 
                                                                                
            while (!count <= 100) do                                            
                do! Async.Sleep(time / 100)                                     
                count := !count + 1                                             
            return result                                                       
            }                                                                   
        { id = id; comp = job; token = token; state = JobState.NotStarted }     
                                                                                
    // The ID of the job whose status changed and the new status.               
    type JobStatusChangedMessage = int * JobState                               
                                                                                
    type JobCollection(numJobs) =                                                                                                                               
        let mutable tokenSources = Array.init numJobs (fun _ -> new CancellationTokenSource()) 

        // The problem is that an array is mutable shared data, so all updates will be        
        // managed by a jobsStatus agent.                                                     
        let mutable jobs = Array.init numJobs (fun id ->                                      
            createJob(id, tokenSources.[id].Token))                                           
                                                                                              
        let mutable parentIds = Array.create numJobs None                                     
                                                                                              
        let jobsStatusAgent = MailboxProcessor<JobStatusChangedMessage>.Start(fun inbox ->    
            let rec loop n =                                                                  
                async {                                                                       
                    let! jobId, jobState = inbox.Receive()                                    
                    jobs.[jobId] <- { jobs.[jobId] with state = jobState }                    
                    do! loop (n + 1)                                                          
                }                                                                             
            loop (0))                                                                         
                                                                                              
        member this.Item                                                                      
            with get(index) =                                                                 
                jobs.[index]                                                                  
            and set index (value : Job) =                                                     
                jobsStatusAgent.Post(index, value.state)                                      
                                                                                              
        member this.TokenSource with get(index) = tokenSources.[index]                        
                                                                                              
        member this.ParentId with get(index) = parentIds.[index]                               

        member this.Length = jobs.Length
                                                                                              

        // Child jobs have the same CancellationToken as the parent job, so                   
        // if you cancel the parent job, the child jobs are also cancelled.                   
        // This function locks the collection during these modifications.                     
        member this.AddChildJobs(numJobs, parentId) =                                         
           lock this (fun () ->                                                               
               let firstNewId = jobs.Length                                                   
               let newJobs = Array.init numJobs (fun id -> createJob(firstNewId + id, jobs.[parentId].token))
               let newTokenSources = Array.create numJobs tokenSources.[parentId]                            
               let newParentIds = Array.create numJobs (Some(parentId))                                      
               jobs <- Array.append jobs newJobs                                                             
               tokenSources <- Array.append tokenSources newTokenSources                                     
               parentIds <- Array.append parentIds newParentIds)                                             
                                                                                                             
                                                                                                             
    let numJobs = 10                                                                                         
    let jobs = new JobCollection(numJobs)                                                                    
                                                                                                             
    jobs.AddChildJobs(4, 2)                                                                                  
                                                                                                             
    let printAgent = MailboxProcessor<string>.Start(fun inbox ->                                             
        let rec loop n =                                                                                     
            async {                                                                                          
                let! str = inbox.Receive()                                                                   
                printfn "%s" str                                                                             
                do! loop (n + 1)                                                                             
            }                                                                                                
        loop (0))                                                                                            
                                                                                                             
    // This agent processes when jobs are completed.                                                         
    let completeAgent = MailboxProcessor<Message>.Start(fun inbox ->                                         
        let rec loop n =                                                                                     
            async {                                                                                          
                let! (id, result) = inbox.Receive()                                                          
                printAgent.Post <| sprintf "The result of job #%d is %f" id result                           
                // Remove from the list of jobs.                                                             
                jobs.[id] <- { jobs.[id] with state = JobState.Completed}                                    
                do! loop (n + 1)                                                                             
            }                                                                                                
        loop (0))                                                                                            
                                                                                                             
    // This agent starts each job in the order it is received.                                               
    let runAgent = MailboxProcessor<Job>.Start(fun inbox ->                                                  
        let rec loop n =                                                                                     
            async {                                                                                          
                let! job = inbox.Receive()                                                                   
                let str = sprintf "Starting job #%d" job.id                                                  
                match jobs.ParentId(job.id) with                                                             
                | Some id -> printAgent.Post <| sprintf "%s with parentId #%d" str id                        
                | None -> printAgent.Post str                                                                
                // Add the new job information to the list of running jobs.                                  
                jobs.[job.id] <- { jobs.[job.id] with state = JobState.Running }                             
                // Start the job.                                                                            
                Async.StartWithContinuations(job.comp,                                                       
                    (fun result -> completeAgent.Post(job.id, result)),                                      
                    (fun _ -> ()),                                                                           
                    (fun cancelException -> printAgent.Post <| sprintf "Canceled job #%d" job.id),           
                    job.token)                                                                               
                do! loop (n + 1)                                                                             
                }                                                                                            
        loop (0))                                                                                            
                                                                                                             
                                                                                                             
    for id in 0 .. jobs.Length - 1 do                                                                        
        runAgent.Post(jobs.[id])                                                                             
                                                                                                             
    let cancelJob(cancelId) =                                                                                
        if (cancelId >= 0 && cancelId < numJobs && jobs.[cancelId].state = JobState.Running) then            
            jobs.[cancelId] <- { jobs.[cancelId] with state = JobState.Canceled }                            
            jobs.TokenSource(cancelId).Cancel()                                                              
            printAgent.Post <| sprintf "Cancelling job #%d" cancelId                                         
        else                                                                                                 
            printAgent.Post <| sprintf "Job #%d could not be canceled." cancelId                             
                                                                                                             
    printAgent.Post <| "Specify a job by number to cancel it, and then press Enter."                         
                                                                                                             
    let mutable finished = false                                                                             
    while not finished do                                                                                    
        let input = System.Console.ReadLine()                                                                
        let a = ref 0                                                                                        
        if (Int32.TryParse(input, a) = true) then                                                            
            cancelJob(!a)                                                                                    
        else                                                                                                 
            printAgent.Post <| "Closing."                                                                    
            finished <- true                                                                                 
                                                                                                             