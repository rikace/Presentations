namespace FRPFSharp

open System
open System
open System.Drawing
open System.Drawing.Imaging 
open System.Windows.Forms

module Primitives =
    // Arguments for evaluating time-varying values
    type Time = { Time : float32 }

    // Single case discriminated union
    // Primitive behavior functions and values
    type Behavior<'a> = 
      // Function evaluates the actual value
      | Behavior of (Time -> 'a)
      
    // Creates a 'Behavior'     
    let create(``time -> a``) = Behavior(``time -> a``)

    /// Creates a constant function returning 'n' at any time
    let constant n = Behavior(fun _ -> n)

    /// Return the current time
    let time = Behavior(fun t -> t.Time)

    /// Time varying value from -1 to 0 (sinusoid)
    let circularAnim = Behavior(fun t -> sin (t.Time * float32 Math.PI))
    
    // Reading values of behaviors at the specified time
    // Extract function value using pattern matching
    let readValue(Behavior ``Time -> a``, time) = 
            // Run the function and return the valuea at given time
            ``Time -> a`` { Time = time }   

    module Behavior = 
  
      // Create behavior that applies 'f' to a returned value
      let map f (Behavior(fv)) = Behavior(fun t -> f (fv t))

      // Lifting functions of multiple arguments
      let lift1 = map
  
      let lift2 f (Behavior(fv1)) (Behavior(fv2)) = 
            Behavior(fun t -> f (fv1(t)) (fv2(t)))
  
      let lift3 f (Behavior(fv1)) (Behavior(fv2)) (Behavior(fv3)) = 
            Behavior(fun t -> f (fv1(t)) (fv2(t)) (fv3(t)))


    type Behavior<'a> with
      static member (+) (a:Behavior<float32>, b) = 
        // Lift the standard operator
        Behavior.lift2 (+) a b
  
      // Multiplication for behaviors of 32bit floats
      static member (*) (a:Behavior<float32>, b) = 
        Behavior.lift2 (*) a b


    // Speeding up and delaying behaviors 
    let wait delay (Behavior(a)) = 
        Behavior(fun t -> a { t with Time = t.Time + delay })

    let faster q (Behavior(a)) = 
        Behavior(fun t -> a { t with Time = t.Time * q })


    let switch (init:Behavior<'a>) (evt:IObservable<Behavior<'a>>) =  
      let current = ref init      
      // Update the behavior
      evt |> Observable.add (fun arg -> current := arg)
      Behavior(fun ctx -> 
        // Get the current behavior and run it
        let (Behavior(f)) = !current
        f(ctx))



module BehaviorModule =

    let lift0(f : 'a -> 'b -> 'c) (a : Behavior<'a>) (b : Behavior<'b>) = Behavior.lift0(f, a, b)
    
    let lift1(f, a : Behavior<_>, b : Behavior<_>, c : Behavior<_>) = Behavior.lift1(f, a, b, c)
    
    let lift2(f, a : Behavior<_>, b : Behavior<_>, c : Behavior<_>, d : Behavior<_>) = a.lift2(f, b, c, d)

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

    let accum  (initState : 'b) (f : 'a -> 'b -> 'b) (evt:Event<_>) : Behavior<'b> = evt.accum(initState,f) 
  
    let once (evt:Event<_>) = evt.Once()

    let send (a:'a) (evt:Event<'a>) = evt.send a
