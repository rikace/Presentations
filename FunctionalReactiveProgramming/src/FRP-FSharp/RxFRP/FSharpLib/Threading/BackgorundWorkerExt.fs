#if INTERACTIVE 
#r "FSharp.PowerPack.dll"
#endif

namespace Easj360FSharp 

open System.ComponentModel
open System.Threading

module BackgroundWorkerExtensions =
  open System
  open System.ComponentModel
  open System.Threading
  open Microsoft.FSharp.Control
 
  type BackgroundWorker with
    member this.AsyncRunWorker(?argument:obj) =
      async { let arg = defaultArg argument null
              this.RunWorkerAsync(arg)
              let! args = Async.AwaitEvent(this.RunWorkerCompleted, cancelAction=this.CancelAsync)
              let result = 
                if args.Cancelled then
                  AsyncCanceled (new OperationCanceledException())
                elif args.Error <> null then AsyncException args.Error
                else  AsyncOk args.Result 
              return! AsyncResult.Commit(result) } 

  type IDelegateEvent<'Del when 'Del :> Delegate > with
    member this.Subscribe(d) =
      this.AddHandler(d)
      { new IDisposable with
          member disp.Dispose() =
            this.RemoveHandler(d) }
 
  let worker = new BackgroundWorker(WorkerSupportsCancellation = true)

  #if INTERACTIVE
  let context = SynchronizationContext.Current
  SynchronizationContext.SetSynchronizationContext(null)
  #endif
 
  let workerResult (worker:BackgroundWorker) =
      async { use e = worker.DoWork.Subscribe(fun _ args -> 
                                                Thread.Sleep(5000)
                                                //args.Result <- sprintf "Hello %A" args.Argument
                                              )
              return! worker.AsyncRunWorker ("Matt") }
      |> Async.RunSynchronously

  workerResult worker |> printfn "%A"
  workerResult worker |> printfn "%A"
  workerResult worker |> printfn "%A"

  #if INTERACTIVE
  SynchronizationContext.SetSynchronizationContext(context)
  #endif
  do ()