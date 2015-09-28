module InMemoryBus


open System
open System.Collections.Generic
open System.Threading
open System.Threading.Tasks
open System.Reactive.Linq
open IBus

type Subject<'a> = Reactive.Subjects.Subject<'a>


// Implementation of <see cref="IBus"/> that keeps publishers and subscriptions in memory.
[<SealedAttribute>]
type InMemoryBus<'a>() =
    let subject = new Subject<'a>()
    let publisherSubscriptions = new List<IDisposable>()

    interface IBus<'a> with 
        // Adds the given <see cref="IObservable{T}"/> as a message source.
        member x.AddPublisher(observable:IObservable<'a>) =
          //if (observable == null) throw new ArgumentNullException("observable");
          publisherSubscriptions.Add(observable.Subscribe(fun msg -> subject.OnNext(msg)))
    

        // Notifies the provider that an observer is to receive notifications.
        // A reference to an interface that allows observers to stop receiving notifications before the provider has finished sending them.
        member x.Subscribe(observer:IObserver<'a>) : IDisposable =
          subject.Subscribe(observer)


        member x.Dispose() =
            publisherSubscriptions.ForEach(fun d -> d.Dispose())
            subject.Dispose()

// Default implementation of <see cref="ISubscriptionService"/>.
[<SealedAttribute>]
type SubscriptionService<'a>(observable:IObservable<'a>) =
    // Creates a new instance of <see cref="SubscriptionService"/>.
    
    interface ISubscriptionService<'a> with
    // Subscribes the given handler to the message bus. Only messages for which the given predicate resolves to true will be passed to the handler.
        member x.Subscribe canHandle handle: IDisposable =            
                observable.Where(canHandle).Subscribe(handle) 

