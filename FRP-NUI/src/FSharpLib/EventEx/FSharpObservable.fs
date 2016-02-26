namespace Easj360FSharp

open System
open System.Diagnostics

module FSharpObservable = 

    type IObserver<'T> =
        abstract OnNext : value : 'T -> unit
        abstract OnError : error : exn -> unit
        abstract OnCompleted : unit -> unit

    type IObservable<'T> =
        abstract Subscribe : observer : IObserver<'T> -> System.IDisposable

    /// Utility class for creating a source of 'serialized' IObserver events.
    type ObservableSource<'T>() =
        let protect f =
            let mutable ok = false
            try
                f()
                ok <- true
            finally
                Debug.Assert(ok, "IObserver methods must not throw!")
        let mutable key = 0
        // Why a Map and not a Dictionary?  Someone's OnNext() may unsubscribe, so
        // we need threadsafe 'snapshots' of subscribers to Seq.iter over
        let mutable subscriptions = Map.empty : Map<int,IObserver<'T>>
        let next(x) =
            subscriptions |> Seq.iter (fun (KeyValue(_,v)) ->
                protect (fun () -> v.OnNext(x)))
        let completed() =
            subscriptions |> Seq.iter (fun (KeyValue(_,v)) ->
                protect (fun () -> v.OnCompleted()))
        let error(e) =
            subscriptions |> Seq.iter (fun (KeyValue(_,v)) ->
                protect (fun () -> v.OnError(e)))
        let thisLock = new obj()
        let obs =
            { new IObservable<'T> with
                member this.Subscribe(o) =
                    let k =
                        lock thisLock (fun () ->
                            let k = key
                            key <- key + 1
                            subscriptions <- subscriptions.Add(k, o)
                            k)
                    { new IDisposable with
                        member this.Dispose() =
                            lock thisLock (fun () ->
                                subscriptions <- subscriptions.Remove(k)) } }
        let mutable finished = false
        // The source ought to call these methods in serialized fashion (from
        // any thread, but serialized and non-reentrant)
        member this.Next(x) =
            Debug.Assert(not finished, "IObserver is already finished")
            next x
        member this.Completed() =
            Debug.Assert(not finished, "IObserver is already finished")
            finished <- true
            completed()
        member this.Error(e) =
            Debug.Assert(not finished, "IObserver is already finished")
            finished <- true
            error e
        // The IObservable object returned here is threadsafe; you can subscribe
        // and unsubscribe (Dispose) concurrently
        member this.AsObservable = obs

