#load "AgentSystem.fs"
open AgentSystem.LAgent
open System
open System.Threading.Tasks

// Agent version
type CounterMessage =
| Add of int
| Print

let counterF = fun msg count ->
    match msg with
    | Add(i)    -> count + i
    | Print     -> printfn "The value is %i" count; count
    
let c1 = spawnAgent counterF 0
c1 <-- Add(3)
c1 <-- Print

// AchtiveObject version
type Counter() =
    let w = new WorkQueue()
    let mutable count = 0
    member c.Add x = w.Queue (fun () -> 
        count <- count + x
        )
    member c.Print () = w.Queue (fun () -> 
        printfn "The value is %i" count
        )
    member c.CountTask = w.QueueWithTask(fun () ->
        count
        )
    member c.CountAsync = w.QueueWithAsync(fun () ->
        count
        )

    
let c = new Counter()
c.Add 3
c.Print
printfn "The count using Task is %i" (c.CountTask.Result)
Async.RunSynchronously (
            async {
                let! count = c.CountAsync
                printfn "The countusing Async is %i" count
            })
System.Threading.Thread.Sleep(500)
    

                      
