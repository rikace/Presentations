module IBus

open System


// Message bus used to distribute messages.
type IBus<'a> =
    inherit IObservable<'a>
    inherit IDisposable

    // Adds the given <see cref="IObservable{T}"/> as a message source.
    abstract AddPublisher : IObservable<'a> -> unit

  // Subscribes an action to the message bus.
type ISubscriptionService<'a> =
    // Subscribes the given handler to the message bus. Only messages for which the given predicate resolves to true will be passed to the handler.
    abstract Subscribe : ('a -> bool) -> ('a -> unit) -> IDisposable


