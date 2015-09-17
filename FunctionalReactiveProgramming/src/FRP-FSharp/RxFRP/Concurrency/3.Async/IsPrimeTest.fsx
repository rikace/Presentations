#load "..\CommonModule.fsx"

open System
open System.Threading
open System.Diagnostics
open Common

// program to determine if various large numbers are prime

module IsPrimeTest =


    let stopWatch = new Stopwatch() 
    let ResetStopWatch() = stopWatch.Reset(); stopWatch.Start() 
    let ShowTime() = printfn "took %d ms" stopWatch.ElapsedMilliseconds 

    let PrimtResults  primeInfo =
        primeInfo
        |> Array.filter (fun (x,b) -> b)  
        |> Array.iter (fun (x,b) -> printf "%d," x)  
        printfn ""          

    // IsPrime : int -> bool 
    let IsPrime x = // slow implementation
        let mutable i = 2 
        let mutable foundFactor = false 
        while not foundFactor && i < x do 
            if x % i = 0 then 
                foundFactor <- true 
            i <- i + 1 
        not foundFactor

    let nums = [| for i in 10000000..10004000 -> i |]

    ResetStopWatch()
    // primeInfo = array<int * bool>    
    let primeInfo = nums   |> Array.map (fun x -> (x,IsPrime x)) 
    ShowTime() 
    PrimtResults primeInfo 

    // THREAD version
    ResetStopWatch() 
    // we need to "join" at the end to know when we’re done, and these will help do that 
    let mutable numRemainingComputations = nums.Length 
    let mre = new ManualResetEvent(false) 
    // primeInfo = array<int * bool>    
    let primeInfo' = Array.create nums.Length (0,false) 
    nums  
    |> Array.iteri (fun i x -> ignore (ThreadPool.QueueUserWorkItem(fun o ->  
                primeInfo'.[i] <- (x, IsPrime x) 
                // if we’re the last one, signal that we’re done 
                if Interlocked.Decrement(&numRemainingComputations) = 0 then 
                    mre.Set() |> ignore)))     
    // wait until all done 
    mre.WaitOne() |> ignore 
    ShowTime()
    PrimtResults primeInfo'


    // ARRAY PARALLEL verision
    ResetStopWatch() 
    let primeInfo'' = nums |> Array.Parallel.map(fun x -> (x, IsPrime x))
    ShowTime()
    PrimtResults primeInfo''

    // ASYNC version
    ResetStopWatch()  
    let primeInfo''' = nums    
                    |> Array.map (fun x -> async { return (x, IsPrime x) } )   
                    |> Async.Parallel   
                    |> Async.RunSynchronously 
    ShowTime()  
    PrimtResults primeInfo''' 