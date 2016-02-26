namespace FsFRPx

open Misc

module FsFRPx = 
    // Reactive
    type Time = float
    
    
        //let (==>) (evt:'a Event) (f:'a -> 'b) : 'b Event = 
        //let (-=>) (evt:'a Event) (b:'b) : 'b Event = ()
        //let untilB :: Behavior a -> Event (Behavior a) -> Behavior a

        (*
        let (==>) (f:a -> b) (g:a -> b) = 
        (==>) :: Event a -> (a -> b) Event b
        f e ==> f = map (map f) >> f e
        *)

                                // applicative functor 
                                //Some common Behaviors
                                // timeB : Time Behavior
                                // deltaTimeB : Time -> Behavior 
                                // switchB : 'a Behavior -> 'a Behavior Event -> 'a Behavior
                                // val untilB : 'a Behavior -> 'a Behavior Event -> 'a Behavior
                                // events
                                // val ( =>> ) : 'a Event -> ('a -> 'b) -> 'b Event
                                // val ( --> ) : 'a Event -> 'b -> 'b Event
                                // val snapshotE : 'a Event -> 'b Behavior -> ('a * 'b) Event
                                // val snapshotBehaviorOnlyE : 'a Event -> 'b Behavior -> 'b Event
                                // val stepB : 'a -> 'a Event -> 'a Behavior
                                // memoization of Behaviors
                                // val stepAccumB : 'a -> ('a -> 'a) Event -> 'a Behavior
    type Behavior<'a> = 
        | Behavior of (Time -> ('a * (unit -> 'a Behavior)))
    
    // val ( .|. ) : 'a Event -> 'a Event -> 'a Event
    // orE : ('a -> 'a -> 'a) -> 'a Event -> 'a Event -> 'a Event
              // val ( .&. ) : 'a Event -> 'b Event -> ('a * 'b) Event
              // val iterE : 'a Event -> 'b list -> ('a * 'b) Event
              // val loopE : 'a Event -> 'b list -> ('a * 'b) Event
              // val whileE : bool Behavior -> unit Event
              // val whenE : bool Behavior -> unit Event
              // val whenBehaviorE : 'a option Behavior -> 'a Event
    type Event<'a> = 
        | Event of (Time -> ('a option * (unit -> 'a Event)))
    
    let atB (Behavior bf) = bf
    let atE (Event bf) = bf
    
    let memoB b = 
        let cache = ref None
        let bref = ref b
        
        let compute b t = 
            let (r, nb) = atB b t
            cache := Some(t, r)
            (r, nb)
        
        let rec bf t = 
            match !cache with
            | Some(t0, r) when t = t0 -> (r, fun () -> resB)
            | Some(t0, r) when t < t0 -> failwith "error"
            | _ -> 
                match compute !bref t with
                | (r, nb) -> 
                    (r, 
                     fun () -> 
                         bref := nb()
                         resB)
        
        and resB = Behavior bf
        resB
    
    let memoE b = 
        let cache = ref None
        let bref = ref b
        
        let compute b t = 
            let (r, nb) = atE b t
            cache := Some(t, r)
            (r, nb)
        
        let rec bf t = 
            match !cache with
            | Some(t0, r) when t = t0 -> (r, fun () -> resB)
            | Some(t0, r) when t < t0 -> failwith "error"
            | _ -> 
                match compute !bref t with
                | (r, nb) -> 
                    (r, 
                     fun () -> 
                         bref := nb()
                         resB)
        
        and resB = Event bf
        resB
    
    let rec pureB f = Behavior(fun t -> (f, fun () -> pureB f))
    
    let rec (<.>) (Behavior f) (Behavior baf) = 
        let rec bf t = 
            let (r, nb) = baf t
            let (rf, nbf) = f t
            (rf r, fun () -> nbf() <.> nb())
        Behavior bf
    
    let rec pureE f = Event(fun t -> (Some f, fun () -> pureE f))
    
    let rec (<..>) (Event f) (Event baf) = 
        let rec bf t = 
            let (r, nb) = baf t
            let (rfo, nbf) = f t
            match rfo with
            | Some rf -> (rf r, fun () -> (nbf()) <..> nb())
            | None -> (None, fun () -> (nbf()) <..> nb())
        Event bf
    
    let rec timeB = Behavior(fun t -> (t, fun () -> timeB))
    
    let deltaTimeB t0 = 
        let rec bf t0 t = (t - t0, fun () -> Behavior(bf t))
        Behavior(bf t0)
    
    let rec switchB b e = 
        let rec bf b e t = 
            let (r, nb) = atB b t
            
            let proc() = 
                let (re, ne) = atE e t
                match re with
                | None -> Behavior(bf (nb()) (ne()))
                | Some newB -> Behavior(bf newB (ne()))
            (r, proc)
        Behavior(bf b e)
    
    let rec untilB b e = 
        let rec bf b e t = 
            let (r, nb) = atB b t
            
            let proc() = 
                let (re, ne) = atE e t
                match re with
                | None -> Behavior(bf (nb()) (ne()))
                | Some newB -> 
                    ne()
                    newB
            (r, proc)
        Behavior(bf b e)
    
    let rec noneE = Event(fun t -> (None, fun () -> noneE))
    
    let rec someE v = Event(fun t -> (Some v, fun () -> someE v))
    
    let (=>>) evt f = 
        let proc evt = 
            match evt with
            | Some evt -> Some(f evt)
            | None -> None
        
        let rec bf evt t = 
            let (r, nevt) = atE evt t
            (proc r, fun () -> (Event(bf (nevt()))))
        
        Event(bf evt)
    
    let rec (-->) ea b = ea =>> (fun _ -> b)
    
    let rec snapshotE evt b = 
        let rec bf evt b t = 
            let (r, nb) = atB b t
            let (re, nevt) = atE evt t
            match re with
            | Some v -> (Some(v, r), fun () -> Event(bf (nevt()) (nb())))
            | None -> (None, fun () -> Event(bf (nevt()) (nb())))
        Event(bf evt b)
    
    let rec snapshotBehaviorOnlyE evt b = snapshotE evt b =>> (fun (_, x) -> x)
    
    let stepB (a : 'a) (evt : 'a Event) = switchB (pureB a) (evt =>> pureB)
    
    let rec stepAccumB a evt = 
        let bf t = 
            let (re, ne) = atE evt t
            match re with
            | Some f -> 
                let na = f a
                (na, fun () -> stepAccumB na (ne()))
            | None -> (a, fun () -> stepAccumB a (ne()))
        Behavior bf
    
    let rec (.|.) ea eb = 
        let proc ra rb = 
            match (ra, rb) with
            | (Some _, _) -> ra
            | (None, Some _) -> rb
            | (None, None) -> None
        
        let bf ea eb t = 
            let (ra, nea) = atE ea t
            let (rb, neb) = atE eb t
            (proc ra rb, fun () -> nea() .|. neb())
        
        Event(bf ea eb)
    
    let orE comp ea eb = 
        let proc ra rb = 
            match (ra, rb) with
            | (Some a, Some b) -> Some(comp a b)
            | (Some _, None) -> ra
            | (None, Some _) -> rb
            | (None, None) -> None
        
        let rec bf ea eb t = 
            let (ra, nea) = atE ea t
            let (rb, neb) = atE eb t
            (proc ra rb, fun () -> Event(bf (nea()) (neb())))
        
        Event(bf ea eb)
    
    let rec (.&.) ea eb = 
        let proc ra rb = 
            match (ra, rb) with
            | (Some va, Some vb) -> Some(va, vb)
            | _ -> None
        
        let bf ea eb t = 
            let (ra, nea) = atE ea t
            let (rb, neb) = atE eb t
            (proc ra rb, fun () -> nea() .&. neb())
        
        Event(bf ea eb)
    
    let rec iterE evt l = 
        match l with
        | [] -> noneE
        | head :: tail -> 
            let bf t = 
                let (re, ne) = atE evt t
                match re with
                | Some v -> (Some(v, head), fun () -> iterE (ne()) tail)
                | None -> (None, fun () -> iterE (ne()) l)
            Event bf
    
    let rec loopE evt l = 
        match l with
        | [] -> noneE
        | head :: tail -> 
            let bf t = 
                let (re, ne) = atE evt t
                match re with
                | Some v -> (Some(v, head), fun () -> loopE (ne()) (tail @ [ head ]))
                | None -> (None, fun () -> loopE (ne()) l)
            Event bf
    
    let rec whileE b = 
        let bf t = 
            let (r, nb) = atB b t
            if r then (Some(), fun () -> whileE (nb()))
            else (None, fun () -> whileE (nb()))
        Event bf
    
    let whenE b = 
        let rec bf b previous t = 
            let (r, nb) = atB b t
            match (previous, r) with
            | (false, true) -> (Some(), fun () -> Event(bf (nb()) r))
            | (false, false) -> (None, fun () -> Event(bf (nb()) r))
            | (true, true) -> (None, fun () -> Event(bf (nb()) r))
            | (true, false) -> (None, fun () -> Event(bf (nb()) r))
        Event(bf b false)
    
    let whenBehaviorE b = 
        let rec bf b previous t = 
            let (r, nb) = atB b t
            match (previous, r) with
            | (None, Some v) -> (Some v, fun () -> Event(bf (nb()) r))
            | (None, None) -> (None, fun () -> Event(bf (nb()) r))
            | (Some v1, Some v2) -> (None, fun () -> Event(bf (nb()) r))
            | (Some v, None) -> (None, fun () -> Event(bf (nb()) r))
        Event(bf b None)
