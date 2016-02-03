namespace FRPFSharp

type BehaviorLoop<'a>(time:Time) =
    let eventLoop = EventLoop<'a>()
    let behavior = new Behavior<'a>(time, eventLoop.Event, Immediate Unchecked.defaultof<_>)

    member this.loop(aOut: Behavior<'a>) : unit =
        eventLoop.loop(aOut.updates())
        behavior.EventValue <- Some (Immediate (aOut.sample()))
