namespace Easj360FSharp
    
    open System
    open System.Threading
    module AsyncGate = 

        type RequestGate(n:int) =
            let sem = new System.Threading.Semaphore(n, n)
            member x.Acquire(?timeout) =
                async { let! ok = Async.AwaitWaitHandle(sem, ?millisecondsTimeout=timeout)
                    if (ok) then
                        return
                            { new System.IDisposable with
                                member x.Dispose() =
                                    sem.Release() |> ignore }
                    else
                        return! failwith "Couldn't acquire Gate" }  
                  
                  