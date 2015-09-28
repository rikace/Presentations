namespace Common

open System
open System.Threading.Tasks
// Source code from: http://theburningmonk.com/2012/10/f-helper-functions-to-convert-between-asyncunit-and-task/

[<AutoOpen>]
module Async =
    let inline awaitPlainTask (task: Task) = 
        // rethrow exception from preceding task if it fauled
        let continuation (t : Task) : unit =
            match t.IsFaulted with
            | true -> raise t.Exception
            | arg -> ()
        task.ContinueWith continuation |> Async.AwaitTask
 
    let inline startAsPlainTask (work : Async<unit>) = Task.Factory.StartNew(fun () -> work |> Async.RunSynchronously)
 

    let inline raise(ex) = Async.FromContinuations(fun (_,econt,_) -> econt ex)

    let inline awaitTask (t: Task) =
            let tcs = new TaskCompletionSource<unit>(TaskContinuationOptions.None)
            t.ContinueWith((fun _ -> 
                if t.IsFaulted then tcs.SetException t.Exception
                elif t.IsCanceled then tcs.SetCanceled()
                else tcs.SetResult(())), TaskContinuationOptions.ExecuteSynchronously) |> ignore
            async {
                try
                    do! Async.AwaitTask tcs.Task
                with
                | :? AggregateException as ex -> 
                    do! raise (ex.Flatten().InnerExceptions |> Seq.head) }

    let inline toTask (async : Async<_>) = 
        Task.Factory.StartNew(fun _ -> Async.RunSynchronously(async))

    let inline toActionTask (async : Async<_>) = 
        Task.Factory.StartNew(new Action(fun () -> Async.RunSynchronously(async) |> ignore))

    /// Helper that can be used for writing CPS-style code that resumes
    /// on the same thread where the operation was started.
    let synchronize f = 
      let ctx = System.Threading.SynchronizationContext.Current 
      f (fun g ->
        let nctx = System.Threading.SynchronizationContext.Current 
        if ctx <> null && ctx <> nctx then ctx.Post((fun _ -> g()), null)
        else g() )