namespace Easj360FSharp

open System
open AgentHelper

module AgentMapReduce =

    //type Agent<'T> = MailboxProcessor<'T>  
    //This is the response the supervisor 
    //gives to the worker request for work 
    type 'work SupervisorResponse = 
    | Work of 'work //a piece of work 
    | NoWork//no work left to do   

    //This is the message to the supervisor 
    type 'work WorkMsg =  
    | ToDo of 'work //piles up work in the Supervisor queue 
    | WorkReq of   AsyncReplyChannel<SupervisorResponse<'work>> //'   
    //The supervisor agent can be interacted with 
    type AgentOperation =  
    | Stop //stop the agent 
    | Status //yield the current status of supervisor  

    type 'work SupervisorMsg =  
    | WorkRel of 'work WorkMsg 
    | Operation of AgentOperation   
    //Supervises Map and Reduce workers 

    module AgentSupervisor =      
        let getNew (name:string) =          
            new Agent<SupervisorMsg<'work>>(fun inbox ->
                let rec loop state  = async {                 
                    let! msg = inbox.Receive()                 
                    match msg with                  
                    | WorkRel(m) ->                      
                            match m with                      
                            | ToDo(work) ->                          
                                    let newState = work::state
                                    return! loop newState                     
                            | WorkReq(replyChannel) ->                           
                                    match state with                          
                                        | [] ->                              
                                            replyChannel.Reply(NoWork)                             
                                            return! loop []                         
                                        | [item] ->                              
                                            replyChannel.Reply(Work(item))                             
                                            return! loop []                         
                                        | (item::remaining) ->                              
                                            replyChannel.Reply(Work(item))                             
                                            return! loop remaining                 
                    | Operation(op) ->                      
                            match op with                      
                            | Status ->                          
                                Console.WriteLine(name+" current Work Queue "+
                                        string (state.Length))                         
                                return! loop state                     
                            | Stop ->                          
                                Console.WriteLine("Stoppped SuperVisor Agent "+name)
                                return()             
                             }             
                loop [] )     
        let stop (agent:Agent<SupervisorMsg<'work>>) = agent.Post(Operation(Stop))     
        let status (agent:Agent<SupervisorMsg<'work>>) =agent.Post(Operation(Status))  
    
    //Code for the workers 
    type 'success WorkOutcome =  
    | Success of 'success 
    | Fail  

    type WorkerMsg =  
    | Start 
    | Stop 
    | Continue  
 
    module AgentWorker =      
        type WorkerSupervisors<'reduce,'work> =          
            { Map:Agent<SupervisorMsg<'work>> ; Reduce:Agent<SupervisorMsg<'reduce>> }      
    
        let stop (agent:Agent<WorkerMsg>) = agent.Post(Stop)     
        let start (agent:Agent<WorkerMsg>) = agent.Start()                                          
                                             agent.Post(Start)      
    
        let getNewMapWorker( map, supervisors:WorkerSupervisors<'reduce,'work>  ) =          
            new Agent<WorkerMsg>(fun inbox ->               
                let rec loop ()  = async {                 
                    let! msg = inbox.Receive()                 
                    match msg with                  
                    | Start ->  inbox.Post(Continue)                            
                                return! loop ()                 
                    | Continue ->  let! supervisorOrder = supervisors.Map.PostAndAsyncReply(fun replyChannel -> WorkRel(WorkReq(replyChannel)))
                                   match supervisorOrder with                      
                                   | Work(work) -> let! res = map work
                                                   match res with                         
                                                   | Success(toReduce) -> supervisors.Reduce.Post(WorkRel(ToDo(toReduce)))
                                                   | Fail -> Console.WriteLine("Map Fail")                             
                                                             supervisors.Map.Post(WorkRel(ToDo(work)))
                                                             inbox.Post(Continue)                    
                                   | NoWork -> inbox.Post(Continue)                             
                                               return! loop ()                 
                    | Stop -> Console.WriteLine("Map worker stopped")                     
                              return ()                 
                    }             
                loop ()  )    
        

        let getNewReduceWorker(reduce,reduceSupervisor:Agent<SupervisorMsg<'work>>) =         
            new Agent<WorkerMsg>(fun inbox ->               
                let rec loop ()  = async {                 
                    let! msg = inbox.Receive()                 
                    match msg with                 
                    | Start ->  inbox.Post(Continue)                            
                                return! loop()                 
                    | Continue -> let! supervisorOrder = reduceSupervisor.PostAndAsyncReply(fun replyChannel -> WorkRel(WorkReq(replyChannel)))
                                  match supervisorOrder with                      
                                  | Work(work) ->
                                        let! res = reduce work                          
                                        match res with                          
                                        | Success(toReduce) -> inbox.Post(Continue)                         
                                        | Fail -> 
                                             Console.WriteLine("ReduceFail")                             
                                             reduceSupervisor.Post(WorkRel(ToDo(work)))                             
                                             inbox.Post(Continue)                     
                                  | NoWork ->   inbox.Post(Continue)                     
                                                return! loop()                 
                    |Stop -> Console.WriteLine("Reduce worker stopped"); 
                             return ()                  
                }            
                loop() )  

    open AgentWorker  

    type MapReduce<'work,'reduce>(  numberMap:int ,                                 
                                    numberReduce: int,                                 
                                    toProcess:'work list,                                  
                                    map:'work->Async<'reduce WorkOutcome>,                                
                                    reduce:'reduce-> Async<unit WorkOutcome>) =   
                                    
            let mapSupervisor= AgentSupervisor.getNew("MapSupervisor")       
            let reduceSupervisor  = AgentSupervisor.getNew("ReduceSupervisor")      
        
            let workerSupervisors = {Map = mapSupervisor ; Reduce = reduceSupervisor }      
        
            let mapWorkers =          
                [for i in 1..numberMap ->              
                    AgentWorker.getNewMapWorker(map,workerSupervisors) ]     

            let reduceWorkers =          
                [for i in 1..numberReduce ->              
                    AgentWorker.getNewReduceWorker(reduce,workerSupervisors.Reduce) ]       

            member this.Start() =          
                //Post work to do         
                toProcess         
                |>List.iter(fun elem -> mapSupervisor.Post( WorkRel(ToDo(elem))))         
                //Start supervisors         
                mapSupervisor.Start()         
                reduceSupervisor.Start()         
                //start workers         
                List.iter( fun mapper -> mapper |>start) mapWorkers          
                List.iter( fun reducer ->reducer|>start) reduceWorkers      
            
            member this.Status() =  (mapSupervisor|>AgentSupervisor.status)                             
                                    (reduceSupervisor|>AgentSupervisor.status)    

            member this.Stop() =     List.map2(fun mapper reducer ->              
                                            mapper |>stop; reducer|>stop) mapWorkers reduceWorkers  
                                        


//Run some tests 
//let map = function (n:int64) -> async{ return Success(n) }   
//let reduce = function (toto: int64) -> async{ return Success() }  
//let mp = MapReduce<int64,int64>( 1,1,[for i in 1L..1000000L->i],map,reduce)  
//mp.Start() 
//mp.Status() 
//mp.Stop()