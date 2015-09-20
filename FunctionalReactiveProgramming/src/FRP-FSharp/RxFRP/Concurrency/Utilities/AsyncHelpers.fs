namespace AsyncHelpers

open System
open System.Threading

[<AutoOpen>]
module Gate = 
    type RequestGate(n : int) =
        let semaphore = new Semaphore(initialCount = n, maximumCount = n)

        member x.AsyncAcquire(?timeout) =
            async {let! ok = Async.AwaitWaitHandle(semaphore,
                                                   ?millisecondsTimeout = timeout)
                   if ok then
                       return
                         {new System.IDisposable with
                             member x.Dispose() =
                                 semaphore.Release() |> ignore}
                   else
                       return! failwith "couldn't acquire a semaphore" }

[<AutoOpen>]
module EventObservable = 
  
  /// Creates an observable that calls the specified function after someone
  /// subscribes to it (useful for waiting using 'let!' when we need to start
  /// operation after 'let!' attaches handler)
  let guard f (e:IObservable<'Args>) =  
    { new IObservable<'Args> with  
        member x.Subscribe(observer) =  
          let rm = e.Subscribe(observer) in f(); rm } 

[<AutoOpen>]
module Extensions = 

  type RequestGate(n:int) =
        let semaphore = new System.Threading.Semaphore(n,n)
        member x.Aquire(?timeout) = 
            async { let! ok = Async.AwaitWaitHandle(semaphore, ?millisecondsTimeout=timeout)
                    if ok then return { new System.IDisposable with
                                            member x.Dispose() =
                                                semaphore.Release() |> ignore }
                    else return! failwith "Handle not aquired" }


  type System.Threading.Semaphore with
        static member Gate(n:int) =
            RequestGate(n)

  /// Ensures that the continuation will be called in the same synchronization
  /// context as where the operation was started
  let synchronize f = 
    let ctx = System.Threading.SynchronizationContext.Current 
    f (fun g arg ->
      let nctx = System.Threading.SynchronizationContext.Current 
      if ctx <> null && ctx <> nctx then ctx.Post((fun _ -> g(arg)), null)
      else g(arg) )

  type Microsoft.FSharp.Control.Async with 
    static member GuardedAwaitObservable (ev1:IObservable<'a>) gfunc =
      synchronize (fun f ->
        Async.FromContinuations((fun (cont,econt,ccont) -> 
          let rec callback = (fun value ->
            remover.Dispose()
            f cont value )
          and remover : IDisposable  = ev1.Subscribe(callback) 
          gfunc() )))

    /// Constructs workflow that triggers the specified event 
    /// on the GUI thread when the wrapped async completes 
    static member WithResult f (a:Async<_>) = async {
        let! res = a
        f res
        return res }
        
    static member AwaitObservable(ev1:IObservable<'a>) =
      synchronize (fun f ->
        Async.FromContinuations((fun (cont,econt,ccont) -> 
          let rec callback = (fun value ->
            remover.Dispose()
            f cont value )
          and remover : IDisposable  = ev1.Subscribe(callback) 
          () )))
  
    static member AwaitObservable(ev1:IObservable<'a>, ev2:IObservable<'b>) = 
      synchronize (fun f ->
        Async.FromContinuations((fun (cont,econt,ccont) -> 
          let rec callback1 = (fun value ->
            remover1.Dispose()
            remover2.Dispose()
            f cont (Choice1Of2(value)) )
          and callback2 = (fun value ->
            remover1.Dispose()
            remover2.Dispose()
            f cont (Choice2Of2(value)) )
          and remover1 : IDisposable  = ev1.Subscribe(callback1) 
          and remover2 : IDisposable  = ev2.Subscribe(callback2) 
          () )))

    static member AwaitObservable(ev1:IObservable<'a>, ev2:IObservable<'b>, ev3:IObservable<'c>) = 
      synchronize (fun f ->
        Async.FromContinuations((fun (cont,econt,ccont) -> 
          let rec callback1 = (fun value ->
            remover1.Dispose()
            remover2.Dispose()
            remover3.Dispose()
            f cont (Choice1Of3(value)) )
          and callback2 = (fun value ->
            remover1.Dispose()
            remover2.Dispose()
            remover3.Dispose()
            f cont (Choice2Of3(value)) )
          and callback3 = (fun value ->
            remover1.Dispose()
            remover2.Dispose()
            remover3.Dispose()
            f cont (Choice3Of3(value)) )
          and remover1 : IDisposable  = ev1.Subscribe(callback1) 
          and remover2 : IDisposable  = ev2.Subscribe(callback2) 
          and remover3 : IDisposable  = ev3.Subscribe(callback3) 
          () )))
