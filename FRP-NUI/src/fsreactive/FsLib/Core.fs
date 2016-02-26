namespace FsFRPLib

open System

module Core = 


    type Time = float
   
    type 'a Behavior = 
        | Behavior of (Time -> ('a * ReactBeh<'a>))
   
    and 'a ReactBeh = unit -> 'a Behavior   

    
    type 'a Event = 
        | Event of (Time -> (Option<'a> * ReactEvent<'a>))

    and 'a ReactEvent = unit -> 'a Event 




    //  val pureBeh : value:'a -> Behavior<'a>
    let rec pureBehavior (value : 'a) = Behavior(fun (t : Time) -> (value, fun () -> pureBehavior value))
    
    
    // val ( <*> ) : Behavior<('a -> 'b)> -> Behavior<'a> -> Behavior<'b>
    // behA :: (Time -> ('a -> 'b) * (unit -> Behavior<('a -> 'b)>))
    // behB :: (Time -> 'a * (unit -> Behavior<'a>)
    let rec (<*>) (Behavior (behA:(Time -> ('a -> 'b) * (unit -> Behavior<('a -> 'b)>)))) 
                            (Behavior (behB:(Time -> 'a * (unit -> Behavior<'a>)))) =

    //let rec (<*>) (Behavior behA) (Behavior behaB) = 
        let behFun (time : Time) = 
            let (value : 'a, newBehavior : unit -> 'a Behavior) = behB time
            let (rFun : 'a -> 'b, nbfun : unit -> ('a -> 'b) Behavior) = behA time
            (rFun value, fun () -> nbfun() <*> newBehavior())
        Behavior behFun

    
    //  val pureEvt : value:'a -> Event<'a>
    let rec pureEvent value = Event(fun (t : Time) -> (Some(value : 'a), fun () -> pureEvent value))
    
    // Behavior Time
    let rec timeBehavior : Time Behavior = Behavior(fun (t : Time) -> (t, fun () -> timeBehavior))

    // Behavior 'a -> Event<Behavior 'a> -> Behavior 'a
    let rec switchBehavior (behavior : 'a Behavior) (event : Event<'a Behavior>) : 'a Behavior= 
        let decomposeBeh (Behavior beh) (time : Time) = beh time
        let toEvent (Event e) = e
        let rec behFun (b : 'b Behavior) (e : 'b Behavior Event) (t : Time) = 
            let (value : 'b, newBeh : unit -> 'b Behavior) = decomposeBeh b t
            
            let compute() = 
                let (beh, newEvt) = toEvent e t
                match beh with
                | None -> Behavior(behFun (newBeh()) (newEvt()))
                | Some newB -> Behavior(behFun newB (newEvt()))
            (value, compute)
        Behavior(behFun behavior event)

    // Behavior 'a -> Event<Behavior 'a> -> Behavior 'a    
    let rec executeBehaviorUntil (b : 'a Behavior) (e : 'a Behavior Event) =
        let toBehavior (Behavior b) (t : Time) = b t 
        let toEvent (Event e) = e
        let rec behFun (b : 'a Behavior) (e : 'a Behavior Event) (t : Time) = 
            let (value : 'b, newBeh : unit -> 'b Behavior) = toBehavior b t
            
            let compute() = 
                let (behOpt : 'a Behavior option, newEvt : unit -> 'a Behavior Event) = toEvent e t
                match behOpt with
                | None -> Behavior(behFun (newBeh()) (newEvt()))
                | Some newB -> 
                    ignore <| newEvt()
                    newB
            (value, compute)
        Behavior(behFun b e)
    
    // Event 'a
    //let rec noneEvent = Event(fun t -> (None, fun () -> noneEvent))    
    
    // 'a -> Event 'a
    let rec someEvent v = Event(fun t -> (Some v, fun () -> someEvent v))

    // 'a Behavior -> 'a Behavior
    let computeBehavior (behavior:Behavior<'a>) = 
        let timeRef = ref None
        let behaviorRef = ref behavior
        let computeBehavior' (b : 'a Behavior) (t : Time) =
            let toBehavior (Behavior b) (t : Time) = b t 
            let (value : 'a, computeNewBehavior : unit -> 'a Behavior) = toBehavior b t
            timeRef := Some(t, value)
            (value, computeNewBehavior)        
        let rec behaviorFun (time : Time) = 
            match !timeRef with
            | Some(t', value) when time = t' -> (value, fun () -> resultingBehavior)
            | _ -> 
                let (value, nb) = computeBehavior' !behaviorRef time
                (value, fun () -> 
                     behaviorRef := nb()
                     resultingBehavior)
        and resultingBehavior = Behavior behaviorFun
        resultingBehavior
    
    // 'a Event -> 'a EVent
    let computeEvent (event : 'a Event) = 
        let timeRef = ref None
        let eventRef = ref event
        let toEvent (Event e) = e
        let computeEvent' (event : 'a Event) (time : Time) = 
            let (value : 'a option, newEvent : unit -> 'a Event) = toEvent event time
            timeRef := Some(time, value)
            (value, newEvent)
        
        let rec eventFun t = 
            match !timeRef with
            | Some(t0, r) when t = t0 -> (r, fun () -> resultingEvent)
            | _ -> 
                let (r, newEvent) = computeEvent' !eventRef t
                (r, 
                 fun () -> 
                     eventRef := newEvent()
                     resultingEvent)
        
        and resultingEvent = Event eventFun
        resultingEvent
    
    // Event<('a option -> 'b option)> -> Event 'a -> Event 'b
    // (Time -> ('a option -> 'b option) option * (unit -> Event<('a option -> 'b option)>))
    // (Time -> 'a option * (unit -> Event<'a>)
    let rec (<**>) (Event (evtA:Time -> ('a option -> 'b option) option * (unit -> Event<('a option -> 'b option)>)))
                   (Event (evtB:Time -> 'a option * (unit -> Event<'a>))) = 
        let rec evtFun (t : Time) = 
            let (value : 'a option, newEvent : unit -> 'a Event) = evtB t
            let (evtFinA : ('a option -> 'b option) option, evtFunB : unit -> ('a option -> 'b option) Event) = evtA t
            match evtFinA with
            | Some(f : 'a option -> 'b option) -> (f value, fun () -> (evtFunB()) <**> newEvent())
            | None -> (None, fun () -> (evtFunB()) <**> newEvent())
        Event evtFun
    
    // Event 'a -> (a -> b) -> Event 'b
    let (=>>) (event : 'a Event) (f:'a -> 'b) = 
        let apply event = 
            match event with
            | Some evt -> Some(f evt)
            | None -> None
        let toEvent (Event e) = e
        let rec compute (evt : 'a Event) (t : Time) = 
            let (value:'a option, newEvt:unit -> Event<'a>) = toEvent evt t
            (apply value, fun () -> (Event(compute (newEvt()))))
        
        Event(compute event)
    
    // 'a -> Event<'a> -> Behavior<'a>
    let accumBehavior (value : 'a) (event : 'a Event) = 
            switchBehavior (pureBehavior value) (event =>> pureBehavior)
    
    // Event 'a -> 'b -> Event 'b
    let (-->) (event : 'a Event) value = event =>> (fun _ -> value)
    
    // Event<'a> -> b:Behavior<'b> -> Event<'a * 'b>
    let rec snapshotEvent (event : 'a Event) (beh:'b Behavior) =
        let decomposeBeh (Behavior beh) (time : Time) = beh time 
        let decomposeEvent (Event e) = e
        let rec compute event beh t = 
            let (valueBeh, newBeh) = decomposeBeh beh t
            let (valueEvt, newEvt) = decomposeEvent event t
            match valueEvt with
            | Some v -> (Some(v, valueBeh), fun () -> Event(compute (newEvt()) (newBeh())))
            | None -> (None, fun () -> Event(compute (newEvt()) (newBeh())))
        Event(compute event beh)
    
    // 'a -> Event<('a -> 'b)> -> Behavior<'a>
    let rec accumMap (value:'a) (evtFun:Event<'a->'b>) : Behavior<'b>= 
        let decomposeEvent (Event e) = e
        let compute (t : Time) = 
            let (valuefun, newEvt) = decomposeEvent evtFun t
            match valuefun with
            | Some f -> 
                let value = f value
                (value, fun () -> accumMap value (newEvt()))
            | None -> (value, fun () -> accumMap value (newEvt()))
        Behavior compute
 
    // Event<'a> -> Event<'a> -> Event<'a>
    let rec (.|.) eventA eventB = 
        let compute valueA valueB = 
            match (valueA, valueB) with
            | (Some _, _) -> valueA
            | (None, Some _) -> valueB
            | (None, None) -> None
        let decomposeEvent (Event e) = e
        let apply eventA eventB t = 
            let (ra, newEventA) = decomposeEvent eventA t
            let (rb, newEventB) = decomposeEvent eventB t
            (compute ra rb, fun () -> newEventA() .|. newEventB())
        
        Event(apply eventA eventB)
    
    // Event<'a> -> Event<'b> -> Event<'a * 'b>
    let rec (.&.) eventA eventB = 
        let compute valueA valueB = 
            match (valueA, valueB) with
            | (Some va, Some vb) -> Some(va, vb)
            | _ -> None
        let decomposeEvent (Event e) = e
        let apply eventA eventB t = 
            let (ra, nea) = decomposeEvent eventA t
            let (rb, neb) = decomposeEvent eventB t
            (compute ra rb, fun () -> nea() .&. neb())
        Event(apply eventA eventB)
    
    // Bevahior bool -> Event<unit>
    let rec whileBehavior (behPred : bool Behavior) =
        let decomposeBeh (Behavior beh) (time : Time) = beh time 
        let compute time = 
            let (value, newBeh) = decomposeBeh behPred time
            if value then (Some(), fun () -> whileBehavior (newBeh()))
            else (None, fun () -> whileBehavior (newBeh()))
        Event compute

    // Bevahior bool -> Event<unit>
    let onceBehavior behPred =
        let decomposeBeh (Behavior beh) (time : Time) = beh time 
        let rec compute (b : bool Behavior) (previous : bool) t = 
            let (value, newBeh) = decomposeBeh b t
            match (previous, value) with
            | (false, true) -> (Some(), fun () -> Event(compute (newBeh()) value))
            | _ -> (None, fun () -> Event(compute (newBeh()) value))
        Event(compute behPred false)

    // (a -> b) -> Event 'a -> Event 'b
    let fmap (f:'a -> 'b) (event : 'a Event) =
        let apply event = 
            match event with
            | Some evt -> Some(f evt)
            | None -> None
        let toEvent (Event e) = e
        let rec compute (evt : 'a Event) (t : Time) = 
            let (value:'a option, newEvt:unit -> Event<'a>) = toEvent evt t
            (apply value, fun () -> (Event(compute (newEvt()))))
        
        Event(compute event)

    // Event 'a -> 'b -> Event 'b
    let update (event : 'a Event) value = fmap (fun _ -> value) event