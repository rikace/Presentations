namespace FsFRPLib

open FsFRPLib.Core
open System

[<AutoOpen>]
module Integration = 
    // 'a -> ('a Behavior * reactBeah)
    let createBehavior value = 
        let mutable value = value
        let rec recBeh = Behavior(fun time -> (value, fun () -> recBeh))
        (recBeh, fun x -> value <- x)
    





    // bindAliasB : 'a Behavior -> ('b Behavior * ('a -> unit)) -> 'a Behavior
    let bindBehaviors xb beh = 
        let rec bf aliasedB t = 
            let (Behavior bbf) = aliasedB
            let (Behavior bbf') = fst beh
            let (r, nb) = bbf t
            (snd beh) r
            let (r', _) = bbf' t
            (r, (fun () -> Behavior(bf (nb()))))
        computeBehavior (Behavior(bf xb))
    
    type NumClass<'a, 'b> = 
        { plus : 'a -> 'a -> 'a
          minus : 'a -> 'a -> 'a
          mult : 'a -> 'b -> 'a
          div : 'a -> 'b -> 'a
          neg : 'a -> 'a }
    
    let floatNumClass = 
        { plus = (+)
          minus = (-)
          mult = (*)
          div = (/)
          neg = (fun (x : float) -> -x) }
    
    // integrateGenB : NumClass<'a, Time> -> 'a Behavior -> Time -> 'a -> 'a Behavior
    let integrateGenB numClass beh time value = 
        let toBehavior (Behavior b) (t : Time) = b t
        
        let rec bf ehb time value t = 
            let (r, nb) = toBehavior beh t
            let i = numClass.plus value (numClass.mult r (t - time))
            (i, fun () -> Behavior(bf (nb()) t i))
        Behavior(bf beh time value)
    
    // integrate : float Behavior -> Time -> float -> float Behavior
    // Integration of numeric Behaviors over time. This is a physical equation
    // that describe the position of a mass under the influence of an accelerating force
    let integrate beh time value = integrateGenB floatNumClass beh time value

    // integrateGenB : NumClass<'a, Time> -> ('a -> 'a) Behavior -> 'a Behavior -> Time -> 'a -> 'a Behavior
    let integrateWithConstraintsGenB numClass constraintsBf b t0 i0 = 
        let toBehavior (Behavior b) (t : Time) = b t
        
        let rec bf constraintsBf b t0 i0 t = 
            let (r, nb) = toBehavior b t
            let i = numClass.plus i0 (numClass.mult r (t - t0))
            let (rcf, ncB) = toBehavior constraintsBf t
            let i' = rcf i
            (i', fun () -> Behavior(bf (ncB()) (nb()) t i'))
        Behavior(bf constraintsBf b t0 i0)
    
    // integrateWithConstraints :  (float -> float) Behavior -> float Behavior -> Time -> float -> float Behavior
    let integrateWithConstraints b t0 i0 = integrateWithConstraintsGenB floatNumClass b t0 i0
    
    // runList : 'a Behavior -> Time list -> 'a list
    let rec runList b l = 
        let toBehavior (Behavior b) (t : Time) = b t
        match l with
        | [] -> []
        | h :: t -> 
            let (r, nb) = toBehavior b h
            r :: runList (nb()) t
    
    // tronB : 'a -> 'b Behavior -> 'b Behavior
    let rec tronB msg b = 
        let toBehavior (Behavior b) (t : Time) = b t
        
        let bf t = 
            let (r, nb) = toBehavior b t
            printf "%A: (t = %f) val = %A \n" msg t r
            (r, fun () -> tronB msg (nb()))
        Behavior bf
    
    // tronE : 'a -> 'b Event -> 'b Event
    let rec tronE msg e = 
        let toEvent (Event e) = e
        
        let bf t = 
            let (r, ne) = toEvent e t
            printf "%A: (t=%f) v=%A \n" msg t r
            (r, fun () -> tronE msg (ne()))
        Event bf
    
    // some constants
    let zeroB = pureBehavior 0.0
    let oneB = pureBehavior 1.0
    let twoB = pureBehavior 2.0
    let piB = pureBehavior Math.PI
    let trueB = pureBehavior true
    let falseB = pureBehavior false
    
    let rec noneB() = Behavior(fun _ -> (None, noneB))
    
    let couple x y = (x, y)
    let coupleB() = pureBehavior couple
    let triple x y z = (x, y, z)
    let tripleB() = pureBehavior triple
    // lifting of classical functions
    let (.*.) (a : Behavior<float>) b = pureBehavior (*) <*> a <*> b
    let (./.) (a : Behavior<float>) b = pureBehavior (/) <*> a <*> b
    let (.+.) (a : Behavior<float>) b = pureBehavior (+) <*> a <*> b
    let (.-.) (a : Behavior<float>) b = pureBehavior (-) <*> a <*> b
    
    let rec negB (a : Behavior<float>) = pureBehavior (fun x -> -x) <*> a
    
    let (.>.) (a : Behavior<_>) b = pureBehavior (>) <*> a <*> b
    let (.<.) (a : Behavior<_>) b = pureBehavior (<) <*> a <*> b
    let (.>=.) (a : Behavior<_>) b = pureBehavior (>=) <*> a <*> b
    let (.<=.) (a : Behavior<_>) b = pureBehavior (<=) <*> a <*> b
    let (.=.) (a : Behavior<_>) b = pureBehavior (=) <*> a <*> b
    let (.<>.) (a : Behavior<_>) b = pureBehavior (<>) <*> a <*> b
    let (.&&.) (a : Behavior<_>) b = pureBehavior (&&) <*> a <*> b
    let (.||.) (a : Behavior<_>) b = pureBehavior (||) <*> a <*> b
    let notB (a : Behavior<_>) = pureBehavior (not) <*> a
    
    type Discontinuity<'a, 'b> = 
        | Disc of ('a Behavior * (Time -> 'a -> 'b) Event * (Time -> 'a -> (Time -> 'a -> 'b) -> Discontinuity<'a, 'b>))
    
    // discontinuityE : Discontinuity<'a, 'b> -> 'a Behavior
    let rec discontinuityE (Disc(xB, predE, bg)) = 
        let evt = 
            snapshotEvent predE ((coupleB() <*> timeBehavior <*> xB)) =>> (fun (e, (t, vb)) -> 
            let disc = bg t vb e
            discontinuityE disc)
        executeBehaviorUntil xB evt
    
    // seqB : 'a Behavior -> 'b behavior -> 'b Behavior
    let rec seqB ba bb = 
        let toBehavior (Behavior b) (t : Time) = b t
        
        let bf t = 
            let (_, na) = toBehavior ba t
            let (b, nb) = toBehavior bb t
            (b, fun () -> seqB (na()) (nb()))
        Behavior bf
    
    // createId : unit -> int
    let createId = 
        let id = ref 0
        fun () -> 
            let i = !id
            id := i + 1
            i
    
    // waitE : Time -> unit Event
//    let rec waitE delta = 
//        let rec bf2 tend t = 
//            if t >= tend then (Some(), fun () -> noneEvent)
//            else (None, fun () -> Event(bf2 tend))
//        
//        let bf t = (None, fun () -> Event(bf2 (t + delta)))
//        Event bf
    
    // startB : (Time -> 'a Behavior) ->  'a Behavior
    let startB fb = 
        let toBehavior (Behavior b) (t : Time) = b t
        
        let bf t = 
            let b = fb t
            toBehavior b t
        Behavior bf
    
    // periodicB : Time -> bool Behavior
//    let rec periodicB period = 
//        let E2 = (someEvent()) =>> (fun () -> periodicB period)
//        let E1 = (waitE period) --> (untillBehavior (pureBehavior true) E2)
//        untillBehavior (pureBehavior false) E1
    
    let someizeBf b = (pureBehavior Some) <*> b
    
    // delayB : 'a Behavior -> 'a -> 'a Behavior
    let delayB b v0 = 
        let toBehavior (Behavior b) (t : Time) = b t
        
        let rec bf b v0 t = 
            let (r, nb) = toBehavior b t
            (v0, fun () -> Behavior(bf (nb()) r))
        Behavior(bf b v0)

module BehaviorTest = 
    type Time = float
    
    type Behavior<'a //type 'a Time = 
                     // Time Behavior     
                     > = 
        | Behavior of (Time -> // float Behavior     
                               (*  the time5B is a little bit more complex to abstract to create a reusable combinator  
        time5B looks very much like to timeB, so we can express it in the same terms

        time5B can be further modified if timeB and the partial application ((*) 5.0) are moved outside of the body of time5B:
        Now, with help of an independent liftBeh function I am able to define a stand alone combinator *)

                               // val liftB : ('a -> 'b) -> 'a Behavior -> 'b Behavior 
                               //    let liftBeh f b =
                               //        let (Behavior bf) = b
                               //        let lbf = fun t -> f (bf t) in Behavior lbf
                               //
                               //    let time5B = liftBeh ((*) 5.0) timeB
                               // transforms a constant into a Behavior 
                               // val constB : 'a -> 'a Behavior 
                               //    let constBeh v =
                               //    let bf t = (v, fun () -> Behavior v) in Behavior bf 
                               //
                               //    let twoB = constBeh 2.
                               //    let helloB = constBeh "Hello FRP in F#!"
                               // Time Behavior     
                               //    let timeB = Behavior (fun t -> t) 
                               //
                               //    // float Behavior     
                               //    let time5B =      
                               //        let (Behavior time (*Time -> Time*)) = timeB (*Time Behavior*)    
                               //        let bf (*Time float*) = fun t -> 5.0 * (time t) in Behavior bf
                               //
                               //    // val liftB : ('a -> 'b) -> 'a Behavior -> 'b Behavior 
                               //    let liftBeh f b =      
                               //        let (Behavior bf (*Time -> ‘a*)) = b (*’a Behavior*)    
                               //        let lbf (*Time ‘b*) = fun t -> f (bf t) in Behavior lbf
                               //
                               //    let time5B (*Float Behavior*)  = liftB ((*) 5.0) timeB
                               // combinators such as liftB allow us to combine Behavior 
                               // and functions in order to create new Behavior. 
                               // However the syntax is still a bit heavy.
                               // The syntax can be simplified quite a lot if we look at the signature of liftB
                               //
                               //    let timeB = Behavior (fun t -> t)  
                               //
                               //    // float Behavior
                               //    let time5B = liftBeh ((*) 5.0) timeB 
                               //    let cosB = liftBeh Math.Sin timeB 
                               //    let cos5B = liftBeh Math.Sin (liftB ((*) 5.0) timeB)
                               //
                               //    let time5 = Behavior (fun t -> t * 5.0)     
                               //    let cosB = Behavior (fun t -> Math.Cos t)     
                               //    let cos5B = Behavior (fun t -> Math.Cos (t * 5.0))
                               'a)
    
    let timeB = Behavior(fun t -> t)
    
    let time5B = 
        let (Behavior time) = timeB
        let bf = fun t -> 5.0 * (time t)
        Behavior bf
