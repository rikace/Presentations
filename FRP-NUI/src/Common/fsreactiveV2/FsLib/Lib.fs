namespace FsFRPx

 open Misc
 
 module Lib = 
 
  open System
  open Misc
  open FsFRPx
 
 
 // runList : 'a Behavior -> Time list -> 'a list

  let rec runList b l  =
    match l with
    |[] -> []
    |h::t -> let (r, nb) = atB b h
             r:: runList (nb()) t
            
 // tronB : 'a -> 'b Behavior -> 'b Behavior

  let rec tronB msg b = 
    let bf t = let (r, nb) = atB b t
               printf "%A: (t = %f) val = %A \n" msg t r
               (r, fun() -> tronB msg (nb()))
    Behavior bf
    
  
 // tronE : 'a -> 'b Event -> 'b Event

  let rec tronE msg e = 
    let bf t = let (r, ne) = atE e t
               printf "%A: (t=%f) v=%A \n" msg t r
               (r, fun() -> tronE msg (ne()))
    Event bf
        
 // some constants
  let zeroB = pureB 0.0
  let oneB = pureB 1.0
  let twoB = pureB 2.0
  
  let piB = pureB Math.PI
  
  let trueB = pureB true
  let falseB = pureB false
 
  let rec noneB() = Behavior (fun _ -> (None, noneB))
 
  let couple x y = (x,y)
  let coupleB() = pureB couple 
  let triple x y z= (x,y,z)
  let tripleB() = pureB triple 
 
  
 // lifting of classical functions
 
  let (.*.) (a:Behavior<float>) b = pureB (*) <.> a <.> b 
  let (./.) (a:Behavior<float>) b = pureB (/) <.> a <.> b 
  let (.+.) (a:Behavior<float>) b = pureB (+) <.> a <.> b 
  let (.-.) (a:Behavior<float>) b = pureB (-) <.> a <.> b 
  
  let rec negB (a:Behavior<float>)  = pureB (fun x -> -x) <.> a
 
  let (.>.) (a:Behavior<_>) b = pureB (>) <.> a <.> b
  let (.<.) (a:Behavior<_>) b = pureB (<) <.> a <.> b
  let (.>=.) (a:Behavior<_>) b = pureB (>=) <.> a <.> b
  let (.<=.) (a:Behavior<_>) b = pureB (<=) <.> a <.> b
  let (.=.) (a:Behavior<_>) b = pureB (=) <.> a <.> b
  let (.<>.) (a:Behavior<_>) b = pureB (<>) <.> a <.> b

  let (.&&.) (a:Behavior<_>) b = pureB (&&) <.> a <.> b
  let (.||.) (a:Behavior<_>) b = pureB (||) <.> a <.> b

  let notB (a:Behavior<_>)  = pureB (not) <.> a 
 
 

  type Discontinuity<'a, 'b> = Disc of ('a Behavior *  (Time -> 'a -> 'b) Event * (Time -> 'a -> (Time -> 'a -> 'b) ->  Discontinuity<'a, 'b>))
    
// discontinuityE : Discontinuity<'a, 'b> -> 'a Behavior

  let rec discontinuityE (Disc (xB, predE, bg))  = 
        let evt = snapshotE predE ((coupleB() <.> timeB <.> xB))  =>>  
                        (fun (e,(t,vb)) -> let disc = bg t vb e
                                           discontinuityE disc)
        untilB xB evt

// seqB : 'a Behavior -> 'b behavior -> 'b Behavior

  let rec seqB ba bb = 
    let bf t = let (_, na) = atB ba t
               let (b, nb) = atB bb t
               (b, fun() -> seqB (na()) (nb()))
    Behavior bf
               

// createId : unit -> int
  let createId =
        let  id = ref  0
        fun () -> let i = !id
                  id := i+1
                  i


// waitE : Time -> unit Event
                  
  let rec waitE delta = 
    let rec bf2 tend t = if t >= tend 
                         then (Some (), fun () -> noneE)
                         else (None, fun () -> Event (bf2 tend))
    let bf t = (None, fun () -> Event (bf2 (t+delta)))
    Event bf

    
// startB : (Time -> 'a Behavior) ->  'a Behavior

  let startB fb = 
        let bf t = let b = fb t
                   atB b t
        Behavior bf

// periodicB : Time -> bool Behavior
 
  let rec periodicB period = 
            let E2 = (someE ()) =>> (fun () -> periodicB period)
            let E1 = (waitE period) --> ( untilB (pureB true) E2)
            untilB (pureB false) E1

  let someizeBf b = (pureB Some) <.> b

// delayB : 'a Behavior -> 'a -> 'a Behavior

  let delayB b v0 = 
        let rec bf b v0 t = let (r, nb) = atB b t
                            (v0, fun () -> Behavior (bf (nb()) r))
        Behavior (bf b v0)
 
 module BehaviorTest = 
    
    type Time = float
    type 'a Behavior = Behavior of (Time -> 'a)
    //type 'a Time = 
    
    // Time Behavior     
    let timeB = Behavior (fun t -> t)
    // float Behavior     
    let time5B =
        let (Behavior time) = timeB
        let bf = fun t -> 5.0 * (time t) in Behavior bf


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
