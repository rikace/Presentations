namespace FRPFSharp

open System
open System.Collections.Generic

type BehaviorValue<'a> = 
    | Delayed of 'a Lazy
    | Immediate of 'a
    member this.Get() : 'a = 
        match this with
        | Delayed l -> l.Value
        | Immediate v -> v

type Time = float

// Implementation using original semantic and approach 

// Time -> 'a Event -> 'a -> 'a Behavior
type Behavior<'a>(time : Time, event : Event<'a>, initValue : 'a BehaviorValue) = 
    let eventValue : 'a BehaviorValue option ref = ref None
    let valueUpdate : 'a option ref = ref None
    let cleanup : Listener option ref = ref None

    do 
        eventValue := Some(initValue)
        Atomically.Run(Handler.New(fun trans1 -> 
                            cleanup := Some(event.Listen(Node.Null, trans1, 
                                                         TransactionHandler.New(fun trans2 a -> 
                                                             if (!valueUpdate).IsNone then 
                                                                 trans2.Last(fun () -> 
                                                                     eventValue 
                                                                     := !valueUpdate |> Option.map (Immediate)
                                                                     //LazyInitValue = null
                                                                     valueUpdate := None)
                                                             valueUpdate := Some a), false))))

    // Time -> 'a -> 'a Behavior
    new(time:Time, initValue : 'a BehaviorValue) = 
            new Behavior<'a>(time, Event<'a>.newDefault(), initValue)

    member this.newValue() = 
        match !valueUpdate with
        | Some v -> v
        | None -> (!eventValue).Value.Get()
    
    member this.sampleNoTrans() = (!eventValue).Value.Get()
    
    member this.EventValue 
        with get () = !eventValue
        and set (value) = eventValue := value

    member this.sample() = Atomically.Apply(fun trans -> this.sampleNoTrans())

    member this.updates() = event    

    member this.Value(trans1 : Atomically) = 
        let out = EventContainer<'a>(fun () -> [| this.sampleNoTrans() |])
        let l = 
            event.Listen(out.Event.Node, trans1, TransactionHandler.New(fun trans2 a -> out.send (trans2, a)), false)
        out.Event.AddCleanup(l).LastFiringOnly(trans1)
    
    member this.Value() = Atomically.Apply(fun trans -> this.Value(trans))

    member this.map<'b> (f : 'a -> 'b) : Behavior<'b> = 
        let f2 : unit -> 'b = fun () -> f (this.sampleNoTrans())
        this.updates().Map(f).HoldLazy(f2)

    member this.lift0<'b, 'c> (f :'a -> 'b -> 'c, b : Behavior<'b>) = 
        let bf : Behavior<'b -> 'c> = this.map (f)
        Behavior.apply (bf, b)
    
    member this.lift1(f, b : Behavior<_>, c : Behavior<_>) = Behavior.apply (Behavior.apply (this.map (f), b), c)
 
    member this.lift2(f, b : Behavior<_>, c : Behavior<_>, d : Behavior<_>) = 
        Behavior.apply (Behavior.apply (Behavior.apply (this.map (f), b), c), d)
    
    member this.collect<'b, 's> (initState : 's, f : 'a -> 's -> ('b * 's)) : Behavior<'b> = 
        Atomically.Run(fun () -> 
            let ea = this.updates().Coalesce(fun fst snd -> snd)
            let ebs = EventLoop<'b * 's>()
            let bbs = ebs.holdLazy (fun () -> f (this.sampleNoTrans()) initState)
            let bs = bbs.map (snd)
            let ebsOut = ea.Snapshot(bs, f)
            ebs.loop (ebsOut)
            bbs.map (fst))
    
    interface IDisposable with
        member this.Dispose() = 
            match !cleanup with
            | Some listener -> listener.Unlisten()
            | None -> ()
    
    static member lift0<'b, 'c> (f : 'a -> 'b -> 'c, a : Behavior<'a>, b : Behavior<'b>) = a.lift0(f, b)
 
    static member lift1 (f, a : Behavior<_>, b : Behavior<_>, c : Behavior<_>) = a.lift1(f, b, c)
   
    static member lift2 (f, a : Behavior<_>, b : Behavior<_>, c : Behavior<_>, d : Behavior<_>) = a.lift2(f, b, c, d)
    
    // ('a -> 'b) Behavior -> 'a Behavior -> 'b Behavior
    static member apply<'b> (bf : Behavior<'a -> 'b>, ba : Behavior<'a>) : Behavior<'b> = 
        let out = EventContainer<'b>.newDefault()
        
        let h = 
            { Fired = false
              Run = Unchecked.defaultof<_> }
        h.Run <- fun (trans1 : Atomically) -> 
            if h.Fired then ()
            else 
                h.Fired <- true
                trans1.Prioritized(out.Event.Node, 
                                   Handler<Atomically>.New(fun (trans2 : Atomically) -> 
                                       out.send (trans2, bf.newValue () (ba.newValue()))
                                       h.Fired <- false
                                       ()))
                ()
        let l1 = bf.updates().Listen_(out.Event.Node, TransactionHandler.New(fun trans1 f -> h.Run trans1))
        let l2 = ba.updates().Listen_(out.Event.Node, TransactionHandler.New(fun trans1 a -> h.Run trans1))
        out.Event.AddCleanup(l1).AddCleanup(l2).HoldLazy(fun () -> bf.sampleNoTrans () (ba.sampleNoTrans()))
    
    static member SwitchB<'b>(bba : Behavior<Behavior<'b>>) : Behavior<'b> = 
        let za = fun () -> bba.sampleNoTrans().sampleNoTrans()
        let out = EventContainer<'b>.newDefault()
        
        let h = 
            { Run = Unchecked.defaultof<_>
              CurrentListener = Unchecked.defaultof<_> }
        h.Run <- fun trans2 (ba : Behavior<'b>) -> 
            h.CurrentListener |> Option.iter (fun l -> l.Unlisten())
            h.CurrentListener <- Some
                                     (ba.Value(trans2)
                                        .Listen(out.Event.Node, trans2, 
                                                TransactionHandler.New(fun trans3 a -> out.send (trans3, a)), false))
            ()
        let l1 = bba.Value().Listen_(out.Event.Node, h)
        out.Event.AddCleanup(l1).HoldLazy(za)
    
    static member SwitchE<'b>(trans1 : Atomically, bea : Behavior<Event<'b>>) : Event<'b> = 
        let out = EventContainer<'b>.newDefault()
        let h2 = TransactionHandler<'b>.New(fun trans2 a -> out.send (trans2, a))
        
        let h1 = 
            { CurrentListener = Some(bea.sampleNoTrans().Listen(out.Event.Node, trans1, h2, false))
              Run = Unchecked.defaultof<_> }
        h1.Run <- fun trans2 (ea : Event<'b>) -> 
            trans2.Last(fun () -> 
                h1.CurrentListener |> Option.iter (fun l -> l.Unlisten())
                h1.CurrentListener <- Some(ea.Listen(out.Event.Node, trans1, h2, false)))
            ()
        let l1 = bea.updates().Listen(out.Event.Node, trans1, h1, false)
        out.Event.AddCleanup(l1)
    
    static member SwitchE<'b>(bea : Behavior<Event<'b>>) : Event<'b> = 
        Atomically.Apply(fun trans -> Behavior<'a>.SwitchE(trans, bea))



and Event<'a>(sampleNow : unit -> obj array) = 
   
    let firings : List<'a> ref = ref (List<_>())
    let listeners : List<TransactionHandler<'a>> ref = ref (List<_>())
    let finalizers : List<Listener> ref = ref (List<_>())
   
    member this.Listen(action : Handler<'a>) : Listener = 
        this.Listen_(Node.Null, TransactionHandler.New(fun trans2 a -> action.Run a))

    member this.Listen(action : 'a -> unit) : Listener = this.Listen(Handler.New(fun a -> action (a)))

    member this.Listen_(target : Node, action : TransactionHandler<'a>) : Listener = 
        Atomically.Apply(fun trans1 -> this.Listen(target, trans1, action, false))

    static member newDefault() : Event<'a> = new Event<'a>(fun () -> null)
    
    member this.Listen(target : Node, trans : Atomically, action : TransactionHandler<'a>, 
                       suppressEarlierFirings : bool) : Listener = 
        lock Atomically.ListenersLock (fun () -> 
            if this.Node.LinkTo(target) then trans.ToRegen <- true
            (!listeners).Add(action))
        trans.Prioritized(target, 
                          Handler.New(fun trans2 -> 
                              let aNow = this.sampleNow()
                              if aNow <> null then 
                                  for t in aNow do
                                      action.Run trans (downcast t)
                              if not suppressEarlierFirings then 
                                  for a in this.Firings do
                                      action.Run trans a))
        Event<'a>.getListener(this, action, target)
    
    member this.AddCleanup(cleanup : Listener) : Event<'a> = 
        (!finalizers).Add(cleanup)
        this
    
    member this.LastFiringOnly(trans : Atomically) : Event<'a> = this.Coalesce(trans, fun first second -> second)
    member this.Node : Node = Node(0L)

    member this.Map<'b>(f : 'a -> 'b) : Event<'b> = 
        let out : EventContainer<'b> = 
            EventContainer<'b>(fun () -> 
                let oi = this.sampleNow()
                if oi <> null then 
                    oi |> Array.map (fun a -> 
                              let result = f (downcast a)
                              upcast result)
                else null)
        
        let l = this.Listen_(out.Event.Node, TransactionHandler.New(fun trans2 a -> out.send (trans2, f a)))
        out.Event.AddCleanup(l)
    
    member this.Hold(initValue : 'a) : Behavior<'a> = 
        Atomically.Apply(fun trans -> new Behavior<'a>(0., this.LastFiringOnly(trans), Immediate initValue))
  
    member this.HoldLazy(initValue : unit -> 'a) : Behavior<'a> = 
        Atomically.Apply
            (fun trans -> new Behavior<'a>(0., this.LastFiringOnly(trans), Delayed(Lazy.Create(initValue))))
    
    member this.Coalesce(trans1 : Atomically, f : 'a -> 'a -> 'a) : Event<'a> = 
        let out = 
            EventContainer<'a>(fun () -> 
                let oi : obj array = this.sampleNow()
                if oi <> null then 
                    let mutable o : 'a = downcast oi.[0]
                    for i in 1..(oi.Length - 1) do
                        o <- f o (downcast oi.[i])
                    Array.singleton (upcast o)
                else null)
        
        let h = Event<'a>.newCoalesceHandler(f, out)
        let l = this.Listen(out.Event.Node, trans1, h, false)
        out.Event.AddCleanup(l)
    
    member this.Coalesce(f : 'a -> 'a -> 'a) : Event<'a> = 
                Atomically.Apply(fun trans -> this.Coalesce(trans, f))
    
    member this.Snapshot<'b, 'c>(b : Behavior<'b>, f : 'a -> 'b -> 'c) : Event<'c> = 
        let out = 
            EventContainer<'c>(fun () -> 
                let oi = this.sampleNow()
                if oi <> null then oi |> Array.map (fun a -> upcast (f (downcast a) (b.sampleNoTrans())))
                else null)
        
        let l = 
            this.Listen_
                (out.Event.Node, TransactionHandler.New(fun trans2 a -> out.send (trans2, f a (b.sampleNoTrans()))))
        out.Event.AddCleanup(l)
    
    member this.Snapshot<'b>(beh : Behavior<'b>) : Event<'b> = this.Snapshot(beh, fun a b -> b)
  
    member this.merge (eb : Event<'a>) : Event<'a> = Event<'a>.merge(this, eb)
    
    member this.Delay() : Event<'a> = 
        let out = EventContainer<'a>.newDefault()
        
        let l1 = 
            this.Listen_(out.Event.Node, 
                         TransactionHandler.New(fun trans a -> 
                             trans.Post(fun () -> 
                                 let trans1 = new Atomically()
                                 try 
                                     out.send (trans1, a)
                                 finally
                                     trans.Close())))
        out.Event.AddCleanup(l1)
    
    member this.Filter(f : 'a -> bool) : Event<'a> = 
        let out = 
            EventContainer<'a>(fun () -> 
                let oi = this.sampleNow()
                if oi <> null then 
                    let oo = oi |> Array.filter (fun a -> f (downcast a))
                    if oo.Length = 0 then null
                    else oo
                else null)
        
        let l = 
            this.Listen_(out.Event.Node, 
                         TransactionHandler.New(fun trans2 a -> 
                             if f a then out.send (trans2, a)))
        
        out.Event.AddCleanup(l)
    
    member this.FilterNotNull() : Event<'a> = this.Filter(fun a -> not (obj.ReferenceEquals(a, null)))
    
    member this.FilterOptional(ev : Event<'a option>) : Event<'a> = 
        let out = 
            EventContainer<'a>(fun () -> 
                let oi = ev.sampleNow()
                if oi <> null then 
                    let oo = oi |> Array.choose (fun c -> downcast c)
                    if oo.Length = 0 then null
                    else oo
                else null)
        
        let l = 
            ev.Listen_
                (out.Event.Node, 
                 TransactionHandler.New(fun trans2 oa -> oa |> Option.iter (fun v -> out.send (trans2, v))))
        out.Event.AddCleanup(l)
    
    member this.Gate(bPred : Behavior<bool>) : Event<'a> = 
        this.Snapshot(bPred, 
                      fun a pred -> 
                          if pred then a
                          else Unchecked.defaultof<_>).FilterNotNull()
    
    member this.Collect<'b, 's>(initState : 's, f : 'a -> 's -> ('b * 's)) : Event<'b> = 
        Atomically.Run(fun () -> 
            let es = EventLoop<'s>()
            let s = es.hold (initState)
            let ebs = this.Snapshot(s, f)
            let eb = ebs.Map(fst)
            let esOut = ebs.Map(snd)
            es.loop (esOut)
            eb)
    
    member this.accum<'s> (initState : 's, f : 'a -> 's -> 's) : Behavior<'s> = 
        Atomically.Run(fun () -> 
            let es = EventLoop<'s>()
            let s = es.hold (initState)
            let esOut = this.Snapshot(s, f)
            es.loop (esOut)
            esOut.Hold(initState))
    
    member this.Once() : Event<'a> = 
        let la : Listener array = Array.zeroCreate 1
        
        let out = 
            EventContainer<'a>(fun () -> 
                let oi = this.sampleNow()
                let mutable oo = oi
                if oo <> null then 
                    if oo.Length > 1 then oo <- [| oi.[0] |]
                    if not (obj.ReferenceEquals(la.[0], null)) then 
                        la.[0].Unlisten()
                        la.[0] <- Unchecked.defaultof<_>
                oo)
        la.[0] <- this.Listen_(out.Event.Node, 
                               TransactionHandler.New(fun trans a -> 
                                   out.send (trans, a)
                                   if not (obj.ReferenceEquals(la.[0], null)) then 
                                       la.[0].Unlisten()
                                       la.[0] <- Unchecked.defaultof<_>))
        out.Event.AddCleanup(la.[0])
    
    member this.sampleNow() : obj array = sampleNow()
  
    member this.Firings = !firings
  
    member this.Listeners = !listeners
    
    member this.send (trans : Atomically, a : 'a) : unit = 
        if this.Firings |> Seq.isEmpty then trans.Last(fun () -> this.Firings.Clear())
        this.Firings.Add(a)
        let clone (l : ICloneable) = l.Clone() :?> TransactionHandler<'a>
        
        let listeners = 
            this.Listeners
            |> Seq.map clone
            |> Seq.toArray
        for action in listeners do
            try 
                action.Run trans a
            with e -> Console.WriteLine(e)
    
    member this.send (a : 'a) : unit = Atomically.Run(Handler.New(fun trans -> this.send (trans, a)))
    
    interface IDisposable with
        member this.Dispose() = 
            for l in !finalizers do
                l.Unlisten()
    
    static member getListener (event : Event<'a>, action : TransactionHandler<'a>, target : Node) : Listener = 
        { Unlisten = 
              fun () -> 
                  lock Atomically.ListenersLock (fun () -> 
                      event.Listeners.Remove(action) |> ignore
                      event.Node.UnlinkTo(target)) }
    
    static member merge (ea : Event<'a>, eb : Event<'a>) : Event<'a> = 
        let out = 
            EventContainer<'a>(fun () -> 
                let oa = ea.sampleNow()
                let ob = eb.sampleNow()
                if oa <> null && ob <> null then Array.append oa ob
                elif oa <> null then oa
                else ob)
        
        let h = TransactionHandler.New(fun trans a -> out.send (trans, a))
        let l1 = ea.Listen_(out.Event.Node, h)
        let l2 = 
            eb.Listen_
                (out.Event.Node, 
                 TransactionHandler.New
                     (fun trans1 a -> 
                     trans1.Prioritized(out.Event.Node, Handler.New(fun trans2 -> out.send (trans2, a)))))
        out.Event.AddCleanup(l1).AddCleanup(l2)
    
    static member mergeWith (f : 'a -> 'a -> 'a, ea : Event<'a>, eb : Event<'a>) : Event<'a> = 
        Event<'a>.merge(ea, eb).Coalesce(f)
   
    static member newCoalesceHandler (f : 'a -> 'a -> 'a, out : EventContainer<'a>) : TransactionHandler<'a> = 
        let mutable accumValid = false
        let mutable accum = Unchecked.defaultof<'a>
        { CurrentListener = None
          Run = 
              fun trans1 a -> 
                  if accumValid then accum <- f accum a
                  else 
                      trans1.Prioritized(out.Event.Node, 
                                         Handler.New(fun trans2 -> 
                                             out.send (trans2, accum)
                                             accumValid <- false
                                             accum <- Unchecked.defaultof<'a>))
                      accum <- a
                      accumValid <- true }








and EventContainer<'a>(sampleNow : unit -> obj array) = 
    let event = new Event<'a>(sampleNow)
    member this.Event : Event<'a> = event
    
    member this.send (trans : Atomically, a : 'a) : unit = 
        if event.Firings |> Seq.isEmpty then trans.Last(fun () -> event.Firings.Clear())
        event.Firings.Add(a)
        let clone (l : ICloneable) = l.Clone() :?> TransactionHandler<'a>
        
        let listeners = 
            event.Listeners
            |> Seq.map clone
            |> Seq.toArray
        for action in listeners do
            try 
                action.Run trans a
            with e -> Console.WriteLine(e)
    
    member this.send (a : 'a) : unit = Atomically.Run(Handler.New(fun trans -> this.send (trans, a)))
   
    static member newDefault() : EventContainer<'a> = EventContainer<'a>(fun () -> null)

and EventLoop<'a>() = 
    let eaOut : Event<'a> option ref = ref None
    
    let event = 
        new Event<'a>(fun () -> 
        match !eaOut with
        | Some e -> e.sampleNow()
        | None -> failwith "EventLoop sampled before it was looped")
    
    let send (trans : Atomically) a = 
        if event.Firings |> Seq.isEmpty then trans.Last(fun () -> event.Firings.Clear())
        event.Firings.Add(a)
        let clone (l : ICloneable) = l.Clone() :?> TransactionHandler<'a>
        
        let listeners = 
            event.Listeners
            |> Seq.map clone
            |> Seq.toArray
        for action in listeners do
            try 
                action.Run trans a
            with e -> Console.WriteLine(e)
    
    do 
        if Atomically.GetCurrentTransaction().IsNone then 
            failwith "EventLoop/BehaviorLoop must be used within an explicit transaction"
    
    member this.Event = event
  
    member this.holdLazy (initValue : unit -> 'a) : Behavior<'a> = event.HoldLazy(initValue)
  
    member this.hold (initValue : 'a) : Behavior<'a> = event.Hold(initValue)
  
    member this.loop (eventOut : Event<'a>) : unit = 
        if (!eaOut).IsSome then failwith "EventLoop looped more than once"
        eaOut := Some eventOut
        event.AddCleanup(eventOut.Listen_(event.Node, TransactionHandler.New(fun trans a -> send trans a))) |> ignore
