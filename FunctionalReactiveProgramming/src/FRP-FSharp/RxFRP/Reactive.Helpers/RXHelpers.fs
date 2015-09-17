namespace Reactive.Helpers

open System
open System.Reactive
open System.Reactive.Subjects
open System.Threading
open System.Threading.Tasks

module RXHelpers =  

    type Task<'a> with
        member t.ToObservale() : IObservable<'a> =
            let subject = new AsyncSubject<'a>()
            t.ContinueWith(fun (f:Task<'a>) ->
            if f.IsFaulted then
                subject.OnError(f.Exception)
            else
                subject.OnNext(f.Result)
                subject.OnCompleted() ) |> ignore
            subject :> IObservable<'a>
    
    type Async<'a> with
        member a.ToObservable() =        
            let subject = new AsyncSubject<'a>()
            Async.StartWithContinuations(a,
                (fun res -> subject.OnNext(res)
                            subject.OnCompleted()),
                (fun exn -> subject.OnError(exn)),
                (fun cnl -> ()))
            subject :> IObservable<'a>

    (async { return 42}).ToObservable().Subscribe(fun x-> printfn "The answer to everything %d" x) 

    let t = Task.Run<int>(new Func<int>(fun () -> 42))
    t.ToObservale().Subscribe(fun x-> printfn "The answer to everything %d" x) 