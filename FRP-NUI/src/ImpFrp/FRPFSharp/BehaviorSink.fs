namespace FRPFSharp

type BehaviorContainer<'a>(time:Time, initValue: 'a) =
    //let initValue = defaultArg initValue Unchecked.defaultof<_>
    let eventSink = EventContainer<'a>.newDefault()
    let behavior = new Behavior<'a>(time, eventSink.Event, Immediate initValue)

    member this.Behavior= behavior

    member this.send(a: 'a) : unit =
        eventSink.send(a)
