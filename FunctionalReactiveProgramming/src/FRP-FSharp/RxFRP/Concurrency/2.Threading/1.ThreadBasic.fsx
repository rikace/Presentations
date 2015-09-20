open System.Threading.Tasks
open System.Threading
open System

let sleep : int -> unit = System.Threading.Thread.Sleep
let print : string -> unit = System.Console.WriteLine

let thread() = 
    [ 1..4 ] |> Seq.iter (fun _ -> 
                    System.Threading.Thread.Sleep 1000
                    printf "Thread ID %d" System.Threading.Thread.CurrentThread.ManagedThreadId)

//let spawn (f : unit -> unit) = 
//    let thread = new Thread(f)
//    thread.Start()
//    thread

let spawn() = 
    let thread = System.Threading.Thread thread // shadowing function
    thread.Start()

spawn()

let threadBody() = 
    for i in 1..5 do
        // Wait 1/10 of a second
        Thread.Sleep(100)
        printfn "[Thread %d] %d..." Thread.CurrentThread.ManagedThreadId i

let spawnThread = new Thread(threadBody)

spawnThread.Start()
ThreadPool.QueueUserWorkItem(fun _ -> 
    for i = 1 to 5 do
        printfn "%d" i)

let spawnPool() = 
    let thread = System.Threading.ThreadPool.QueueUserWorkItem(fun _ -> thread())
    thread

spawnPool()

// Our thread pool task, note that the delegate's
// parameter is of type obj
let printNumbers (max : obj) = 
    for i = 1 to (max :?> int) do
        printfn "%d" i

ThreadPool.QueueUserWorkItem(new WaitCallback(printNumbers), box 5)

//let runMe() = 
//    for i in 1 .. 10 do
//        try
//            Thread.Sleep(1000)
//        with
//            | :? System.Threading.ThreadAbortException as ex -> printfn "Exception %A" ex
//        printfn "I'm still running..."
//
//let createThread() =
//    let thread = new Thread(runMe)
//    thread.Start()
//
//createThread()
//createThread()
let runMe (arg : obj) = 
    for i in 1..10 do
        try 
            Thread.Sleep(1000)
        with :? System.Threading.ThreadAbortException as ex -> printfn "Exception %A" ex
        printfn "%A still running..." arg

ThreadPool.QueueUserWorkItem(new WaitCallback(runMe), "One")
ThreadPool.QueueUserWorkItem(new WaitCallback(runMe), "Two")
ThreadPool.QueueUserWorkItem(new WaitCallback(runMe), "Three")


