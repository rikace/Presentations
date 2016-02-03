namespace FRPFSharp

open System


module BehaviorModule =

    let lift (f : 'a -> 'b -> 'c) (a : Behavior<'a>) (b : Behavior<'b>) = Behavior.lift(f, a, b)

    // ('a -> 'b) Behavior -> 'a Behavior -> 'b Behavior
    let apply(bf : Behavior<'a -> 'b>) (ba : Behavior<'a>) = Behavior.apply(bf, ba) 

    let switchB (bba : Behavior<Behavior<'b>>) : Behavior<'b> = Behavior.SwitchB(bba)

    let swithcE (trans1 : Atomically) (bea : Behavior<Event<'b>>) = Behavior.SwitchE(trans1, bea)


module EventModule = 


    let merge(ae : Event<'a>) (be : Event<'a>) : Event<'a> = Event<'a>.merge(ae, be)

    let map (f : 'a -> 'b) (evt:Event<'a>) : Event<'b> = evt.Map(f)

    let filter (f : 'a -> bool) (evt:Event<'a>) = evt.Filter(f)
    
    let hold (initValue : 'a) (evt:Event<'a>) = evt.Hold(initValue : 'a) 

    let snapShot (b : Behavior<'b>) (f : 'a -> 'b -> 'c) (evt:Event<_>) = evt.Snapshot(b,f)
    
    let snapShotB (b : Behavior<'b>) (evt:Event<_>) = evt.Snapshot(b)

    let accum  (initState : 's) (f : 'a -> 's -> 's) (evt:Event<_>) : Behavior<'s> = evt.accum(initState,f) 
  
    let once (evt:Event<_>) = evt.Once()

    let send (a:'a) (evt:Event<'a>) = evt.send a
