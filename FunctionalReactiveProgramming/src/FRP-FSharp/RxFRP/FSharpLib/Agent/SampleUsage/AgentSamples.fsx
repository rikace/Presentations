#load "AgentSystem.fs"
open AgentSystem.LAgent
open System
open System.Threading
open System.Reflection

// A worker is an agent that doesn't keep a private state 
let echo = spawnWorker (fun s -> printfn "%s" s)
echo <-- "Hello guys!"

// Given that it is stateless, multiple identical ones can process messages at the same time (look at last parameter)
let parallelEcho = spawnParallelWorker (fun s -> printfn "%s" s) 10
parallelEcho <-- "Hello guys!"

// The difference is easily seen by printing out the thread the message is processed on (damn I cannot reuse variable names)
let tprint s = printfn "%s running on thread %i" s Thread.CurrentThread.ManagedThreadId
let echo1 = spawnWorker (fun s -> tprint s)
// The Sleep instruction is helpful in really running the workers on multiple threads (if it's too fast the thread pool reuses the same thread)
let parallelEcho1 = spawnParallelWorker(fun s -> tprint s; Thread.Sleep(300)) 10

// All the msg processing runs on a single thread
let messages = ["a";"b";"c";"d";"e";"f";"g";"h";"i";"l";"m";"n";"o";"p";"q";"r";"s";"t";"b";"c";"d";"e";"f";"g";"h";"i";"l";"m";"n";"o";"p";"q";"r";"s";"t"]
messages |> Seq.iter (fun msg -> echo1 <-- msg)
// It now uses multiple threads (max of 10 as from last parameter to spawnParallelWorker)
messages |> Seq.iter (fun msg -> parallelEcho1 <-- msg)

// Agents are like workers, but they keep an internal state
let counter = spawnAgent (fun msg state -> state + msg) 0

// Now printing out stuff
let counter1 = spawnAgent (fun msg state -> printfn "From %i to %i" state (state + msg); state + msg) 0
counter1 <-- 3 
counter1 <-- 4

// Ohh, and you can restart agents (wasn't this supposed to take ints?? :-) )                         
counter1 <-! Restart
counter1 <-- 3

// If you try to send the wrong type of message, it won't even compile, as below
// counter1 <-- "fst"

// If there is an error in the message processing, by default you get notified with eprintfn with as much info as possible about the error
let errorProneCounter = spawnAgent (fun i s -> printfn "The state was %i and the result was %i" s (s / i); s / i) 100
errorProneCounter <-- 20
errorProneCounter <-- 0

// Optionally, you can name your agents to get a better error message ...
errorProneCounter <-! SetName("Bob")
errorProneCounter <-- 0

// The important thing is that the agent keeps running (keeping its current state)...
errorProneCounter <-- 2

// I want to change the default error behavior for this agent. When an error is generated I want to restart the agent instead ... (you need to CTR+ENTER next two lines together)
let manager = spawnWorker (fun (agent, name, ex, msg, state, initialState) -> printfn "%s restarting ..." name; agent <-! Restart)
errorProneCounter <-! SetManager(manager)

// Generate an error and check that the counter restarted from zero
errorProneCounter <-- 0
errorProneCounter <-- 2

// Also, I might want to do something if the agent dosn't get a message after some time. IE printing stuff to the screen (btw nice way to create a timer)
counter1 <-! SetTimeoutHandler(1000, fun state -> printfn "I'm still waiting for a message in state %A, come on ..." state; ContinueProcessing(state))             
// Still processing messages
counter1 <-- 2
// Getting back to normality
counter1 <-! SetTimeoutHandler(-1,  fun state -> ContinueProcessing(state))
// Or maybe I want to restart every time 1 sec passes ...
counter1 <-! SetTimeoutHandler( 1000, fun state -> printfn "Restart from state %A" state; RestartProcessing)
counter1 <-- 2
// Or maybe I want to stop processing
counter1 <-! SetTimeoutHandler(1000, fun state -> printfn "Restart from state %A" state; StopProcessing)
counter1 <-- 2

// I cannot restart a stopped agent, I need to create a new one (is that a problem?)
let counter2 = spawnAgent (fun msg state -> printfn "From %i to %i" state (state + msg); state + msg) 0
counter2 <-- 2
                                         
// But now my boss tells me that it is all wrong, in my company '+' means '*'. Can I change my agent WITHOUT stopping it (aka keeping its state)?
counter2 <-! SetAgentHandler(fun msg state -> printfn "From %i to %i via multiplication" state (state * msg); msg * state)
counter2 <-- 3

// The above is all that is needed to do hot swapping of code, for example ...
let assemblyNameFromSomewhere, typeNameFromSomewhere, methodNameFromSomewhere = "mscorlib.dll", "System.Console", "WriteLine"
let a = Assembly.Load(assemblyNameFromSomewhere)
let c = a.GetType(typeNameFromSomewhere)
let m = c.GetMethod(methodNameFromSomewhere, [|"".GetType()|])
let newF = fun (msg:int) (state:int) -> m.Invoke(null, [| ("This integer is " + msg.ToString()) |]) |> ignore; msg

// Notice that I also changed the type of the message that the agent can process ...
counter2 <-! SetAgentHandler(newF)
counter2 <-- 2

// I can also change the handler for a worker
echo <-! SetWorkerHandler(fun msg -> printfn "I'm an echo and I say: %s" msg)
echo <-- "Hello"

// Or a parallel worker
parallelEcho <-! SetWorkerHandler(fun msg -> tprint ("I'm new and " + msg))
messages |> Seq.iter (fun msg -> parallelEcho <-- msg;Thread.Sleep(300))

// We can make agents talk etc... (you need to CTRL+ENTER all these lines together)
let wife = spawnWorker (fun msg -> printfn "Wife says: screw you and your '%s'" msg)
let husband = spawnWorker (fun (To, msg) -> printfn "Husband says: %s" msg; To <-- msg)
husband <-- (wife, "Hello")
husband <-- (wife, "But darling ...")
husband <-- (wife, "ok")
