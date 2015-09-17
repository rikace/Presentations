namespace FSharpWcfServiceApplicationTemplate

open System
open System.Threading

    type FSharpAsyncResult(callback:AsyncCallback, state:obj) =
        let mutable m_AsyncCallback = callback  
        let mutable m_State = state
        [<DefaultValue>]
        val mutable m_ManualResetEvent : ManualResetEvent
        [<DefaultValue>]
        val mutable m_IsDisposed : bool  

        interface IAsyncResult with    
            member x.AsyncState
                    with get() =
                            m_State

            member x.AsyncWaitHandle 
                    with get() = 
                            x.m_ManualResetEvent :> System.Threading.WaitHandle


            member x.CompletedSynchronously
                    with get() = 
                            false

            member x.IsCompleted
                    with get() =
                        x.m_ManualResetEvent.WaitOne(0, false)

        interface IDisposable with
            member x.Dispose() =
                    if not x.m_IsDisposed then
                        x.Dispose(true)
                        System.GC.SuppressFinalize(x)
    
        member x.OnCompleted() =    
                if x.m_ManualResetEvent = null then
                    x.m_ManualResetEvent <- new ManualResetEvent(false)
               
                x.m_ManualResetEvent.Set() |> ignore
                if m_AsyncCallback <> null then
                    m_AsyncCallback.Invoke(x)

        member x.Dispose(disposing:bool) =
            try
                if not disposing then
                    x.m_ManualResetEvent.Close()
                    x.m_ManualResetEvent <- null;
                    m_State <- null;
                    m_AsyncCallback <- null;
            finally
                 x.m_IsDisposed <- true