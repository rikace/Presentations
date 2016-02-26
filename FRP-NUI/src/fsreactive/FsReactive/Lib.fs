#light

namespace FsReactive

 open Misc
 
 module Lib = 
 
  open System
  open Misc
  open FsReactive
 
 
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
    Beh bf
    
  
 // tronE : 'a -> 'b Event -> 'b Event

  let rec tronE msg e = 
    let bf t = let (r, ne) = atE e t
               printf "%A: (t=%f) v=%A \n" msg t r
               (r, fun() -> tronE msg (ne()))
    Evt bf
        
 // some constants
  let zeroB = pureB 0.0
  let oneB = pureB 1.0
  let twoB = pureB 2.0
  
  let piB = pureB Math.PI
  
  let trueB = pureB true
  let falseB = pureB false
 
  let rec noneB() = Beh (fun _ -> (None, noneB))
 
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
    Beh bf
               

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
                         else (None, fun () -> Evt (bf2 tend))
    let bf t = (None, fun () -> Evt (bf2 (t+delta)))
    Evt bf

    
// startB : (Time -> 'a Behavior) ->  'a Behavior

  let startB fb = 
        let bf t = let b = fb t
                   atB b t
        Beh bf

// periodicB : Time -> bool Behavior
 
  let rec periodicB period = 
            let E2 = (someE ()) =>> (fun () -> periodicB period)
            let E1 = (waitE period) --> ( untilB (pureB true) E2)
            untilB (pureB false) E1

  let someizeBf b = (pureB Some) <.> b

// delayB : 'a Behavior -> 'a -> 'a Behavior

  let delayB b v0 = 
        let rec bf b v0 t = let (r, nb) = atB b t
                            (v0, fun () -> Beh (bf (nb()) r))
        Beh (bf b v0)
 