using System;
using System.Collections.Generic;
using System.Reactive.Concurrency;
using System.Reactive.Linq;
using System.Reactive.Subjects;

namespace RxPublisherSubscriber
{
    //  Reactive Publisher Subscriber in C#
    public class RxPubSub<T> : IDisposable, ISubject<T>
    {
        private ISubject<T> subject; //#A
        private readonly Func<T, bool> filter; 
        private List<IObserver<T>> observers = new List<IObserver<T>>(); //#B
        private List<IDisposable> observables = new List<IDisposable>(); //#C

        public RxPubSub(ISubject<T> subject, Func<T, bool> filter = null)
        {
            this.subject = subject; //#D
            this.filter = filter ?? new Func<T, bool>(_ => true);
        }
        public RxPubSub() : this(new Subject<T>()) { } //#D

        public IDisposable Subscribe(IObserver<T> observer)
        {
            observers.Add(observer);
            subject.Subscribe(observer);
            return new ObserverHandler<T>(observer, observers); //#E
        }

        public IDisposable AddPublisher(IObservable<T> observable) =>
            observable.SubscribeOn(TaskPoolScheduler.Default).Subscribe(subject); //#F

        public IObservable<T> AsObservable() => 
                subject.AsObservable().Where(filter); //#G

        public void Dispose()
        {
            observers.ForEach(x => x.OnCompleted());
            observers.Clear(); //#H
        }

        public void OnNext(T value) => subject.OnNext(value);
        public void OnError(Exception error) => subject.OnError(error);

        public void OnCompleted() => subject.OnCompleted();
    }

    class ObserverHandler<T> : IDisposable //#I
    {
        private IObserver<T> observer;
        private List<IObserver<T>> observers;

        public ObserverHandler(IObserver<T> observer, List<IObserver<T>> observers)
        {
            this.observer = observer;
            this.observers = observers;
        }

        public void Dispose() //#I
        {
            observer.OnCompleted();
            observers.Remove(observer);
        }
    }
}
