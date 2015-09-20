namespace Easj360FSharp


module AsyncResultFSharpLibrary =

    open System
    open System.Net
    open System.Net.Mail
    open System.Threading

    type internal AsyncResultNoResult(callback: AsyncCallback, state: obj) = 
        let statePending = 0
        let stateCompletedSync = 1
        let stateCompletedAsync = 2
        let mutable completedState = statePending
        let mutable waitHandle: ManualResetEvent = null
        let mutable resultException: exn = null
        interface IAsyncResult with
            member x.AsyncState = state
            member x.AsyncWaitHandle = 
                if waitHandle = null then
                    let isDone = (x :> IAsyncResult).IsCompleted
                    let mre = new ManualResetEvent(isDone)
                    if Interlocked.CompareExchange(&waitHandle, mre, null) <> null
                        then mre.Close()
                        else
                            if not isDone && (x :> IAsyncResult).IsCompleted
                                then waitHandle.Set() |> ignore
                upcast waitHandle
            member x.CompletedSynchronously = 
                Thread.VolatileRead(&completedState) = stateCompletedSync
            member x.IsCompleted = 
                Thread.VolatileRead(&completedState) <> statePending
        member x.SetAsCompleted(e: exn, completedSynchronously: bool) = 
            resultException <- e           
            let prevState = Interlocked.Exchange(&completedState, if completedSynchronously then stateCompletedSync else stateCompletedAsync)
            if prevState <> statePending
                then raise <| InvalidOperationException("You can set a result only once")
            if waitHandle <> null
                then waitHandle.Set() |> ignore
            if callback <> null
                then callback.Invoke(x)
        member x.EndInvoke() = 
            let this = x :> IAsyncResult
            if not this.IsCompleted then
                this.AsyncWaitHandle.WaitOne() |> ignore
                this.AsyncWaitHandle.Close()
                waitHandle <- null
            if resultException <> null
                then raise resultException

    type internal AsyncResult<'a>(callback: AsyncCallback, state: obj) =
        inherit AsyncResultNoResult(callback, state) 
        [<DefaultValue>] val mutable aresult : 'a
        member x.Result 
            with get() = x.aresult
            and set(value) = x.aresult <- value        
        member x.SetAsCompleted(result: 'a, completedSynchronously: bool) = 
                x.aresult <- result
                base.SetAsCompleted(null,completedSynchronously)        
        member x.EndInvoke() =
            base.EndInvoke()
            x.Result  
      