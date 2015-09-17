namespace Easj360FSharp

open System
open System.Threading
open System.Threading.Tasks

module AsyncCancellation = 
    
    let longTaskCTS = new CancellationTokenSource()

    let longRunningTask() =
        let mutable i = 1
        let mutable loop = true
    
        while i <= 10 && loop do
            printfn "%d..." i
            i <- i + 1
            Thread.Sleep(1000)
        
            // Check if the task was cancelled
            if longTaskCTS.IsCancellationRequested then
                printfn "Cancelled; stopping early."
                loop <- false
            
        printfn "Complete!"

    let startLongRunningTask() =
        Task.Factory.StartNew(longRunningTask, longTaskCTS.Token)

    let t = startLongRunningTask()

    // ...    

    longTaskCTS.Cancel()             

/////////////////////

    let asyncTaskX = async { failwith "error" }
 
    asyncTaskX
    |> Async.Catch 
    |> Async.RunSynchronously
    |> function 
       | Choice1Of2 result     -> printfn "Async operation completed: %A" result
       | Choice2Of2 (ex : exn) -> printfn "Exception thrown: %s" ex.Message
    let cancelableTask =
        async {
            printfn "Waiting 10 seconds..."
            for i = 1 to 10 do 
                printfn "%d..." i
                do! Async.Sleep(1000)
            printfn "Finished!"
        }


// Callback used when the operation is canceled
    let cancelHandler (ex : OperationCanceledException) = 
        printfn "The task has been canceled."
 
    Async.TryCancelled(cancelableTask, cancelHandler)
    |> Async.Start
 
    Async.CancelDefaultToken()
 
    let superAwesomeAsyncTask = async { return 42 }

    Async.StartWithContinuations(
        superAwesomeAsyncTask,
        (fun (result : int) -> printfn "Task was completed with result %d" result),
        (fun (exn : Exception) -> printfn "threw exception"),
        (fun (oce : OperationCanceledException) -> printfn "OCE")
    )
    // Callback used when the operation is canceled
//    let cancelHandler (ex : OperationCanceledException) = 
//        printfn "The task has been canceled."
 
    let computation = Async.TryCancelled(cancelableTask, cancelHandler)
    let cancellationSource = new CancellationTokenSource()
 
    Async.Start(computation, cancellationSource.Token)
 
    cancellationSource.Cancel()
