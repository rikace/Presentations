namespace StockTicker.Server.Common

open System
open System.Collections.Generic
open System.Reactive.Concurrency
open System.Reactive.Linq
open System.Reactive.Subjects

type ObserverHandler<'a> (observer:IObserver<'a>, observers: List<IObserver<'a>>) =
    interface IDisposable with
        override this.Dispose() =
            observer.OnCompleted()
            observers.Remove(observer) |> ignore

type RxPubSub<'a> (subject:ISubject<'a>) =
    let observers = List<IObserver<'a>>()

    new() = new RxPubSub<'a>(new Subject<'a>())

    member this.Subscribe(observer:IObserver<'a>) =
        observers.Add(observer)
        subject.Subscribe(observer)  |> ignore
        new ObserverHandler<'a>(observer, observers) :> IDisposable

    member this.AddPublisher(observable: IObservable<'a> ) =
        observable.SubscribeOn(TaskPoolScheduler.Default).Subscribe(subject)

    member this.AsObservable() =
        subject.AsObservable()

    interface IDisposable with
        override this.Dispose() =
            observers.ForEach(fun x -> x.OnCompleted())
            observers.Clear()