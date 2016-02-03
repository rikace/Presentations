#indent "off"

//namespace FrpLib
//
//
//
//map : (a -> b) -> Signal a -> Signal b
//
//filter : (a -> Bool) -> a -> Signal a -> Signal a
//
//merge : Signal a -> Signal a -> Signal a
//
//foldp : (a -> s -> s) -> s -> Signal a -> Signal s
//
//
//Signal.map show Mouse.position  // http://elm-lang.org/examples/mouse-position
//Signal.map show Mouse.isDown    // http://elm-lang.org/examples/mouse-is-down
//
//
//  Signal.map show countClick
//
//
//countClick : Signal Int
//countClick =
//  Signal.foldp (\clk count -> count + 1) 0 Mouse.clicks

open System


    // Considering time as a float
    type Time = float

    // Behavior represents a value that vary with time in a continuous way
    //type 'a Behavior = Time -> 'a

    // Behavior represents a value that vary with time in a continuous way
    type 'a Behavior = Behavior of (Time -> 'a)

    // The constant 
    let two = Behavior (fun _ -> 2.0) 
    let hello = Behavior (fun _ -> "Hello FRP in F#!")
    
    // The time itself
    let time = Behavior (fun t -> t) 	
    let time = Behavior (id) 
        
    // Math on Time
    let time5 = Behavior (fun t -> t * 5.0)    
    let cosB = Behavior (fun t -> Math.Cos t)    
    let cos5B = Behavior (fun t -> Math.Cos (t * 5.0)) 


    // transforms a constant into a Behavior
    // val constB : 'a -> 'a Behavior
    let constB v = 
        let bf t = v in Behavior bf

    let oneB = constB 1
    let helloB = constB "Hello FRP in F#!"


    // the time5B is a little bit more complex to abstarct
    // time5B looks very much like to timeB
    // we can express time5B in terms of timeB
    
    let timeB = Behavior (fun t -> t) 
    // float Behavior    
    let time5B = 
        let (Behavior time (*Time -> Time*)) = timeB (*Time Behavior*)
        let bf (*Time -> float*)= fun t -> 5.0 * (time t)
            in Behavior bf


    //Now, liftB is independent from the rest and can be defined as a stand alone combinator:
    // val liftB : ('a -> 'b) -> 'a Behavior -> 'b Behavior
    let liftBeh f b = 
        let (Behavior bf) = b
        let lbf = fun t -> f (bf t)
                        in Behavior lbf

    // val liftB : ('a -> 'b) -> 'a Behavior -> 'b Behavior
    let liftBeh' f b = 
        let (Behavior bf (*Time -> 'a*)) = b (*'a Behavior*)
        let lbf (*Time 'b*) = fun t -> f (bf t)
               in Behavior lbf

//time5B can be further modified if timeB and the partial application ((*) 5.0) are moved outside of the body of time3B:
    // float Behavior
    let time5B = liftBeh ((*) 5.0) timeB
    

    let timeB = Behavior (fun t -> t) 

    let time5B = liftBeh ((*) 5.0) timeB
    let cosB = liftBeh Math.Sin timeB
    let cos5B = liftBeh Math.Sin (liftBeh ((*) 5.0) timeB)

    // float Behavior -> float Behavior
    let sinF = liftBeh Math.Sin
    // float Behavior
    let sinB = sinF timeB
    // float Behavior -> float Behavior
    let tripleF = liftBeh ((*) 5.0)
    // float Behavior
    let sin3B = sinF (tripleF timeB)


    //val lift2B : ('a -> 'b -> 'c) -> 'a Behavior -> 'b Behavior -> 'c Behavior

    let lift2Beh f b1 b2 = 
        let (Behavior bf1) = b1
        let (Behavior bf2) = b2
        let nbf t = f (bf1 t) (bf2 t) in Behavior nbf

//val lift3B : ('a -> 'b -> 'c -> 'd) -> 'a Behavior -> 'b Behavior -> 'c Behavior -> 'd Behavior

    let lift3Beh f b1 b2 b3 = 
        let (Behavior bf1) = b1
        let (Behavior bf2) = b2
        let (Behavior bf3) = b3
        let nbf  t = f (bf1 t) (bf2 t) (bf3 t)
                       in Behavior nbf

    //Here are some examples of lifted functions.

    // val ( .* ) : (int Behavior -> int Behavior -> int Behavior)
    let (.*) = lift2Beh (*)
    
    // val ( ./ ) : (int Behavior -> int Behavior -> int Behavior)
    let (./) = lift2Beh (/)

// val mapB : ('a -> 'b) Behavior -> 'a list Behavior -> 'b list Behavior

    let mapB f b = (lift2Beh List.map) f b


// val runOne : 'a Behavior -> Time -> 'a
    let runOne b t = let (Behavior bf) = b
                        in bf t

    // val runList : 'a Behavior -> Time list -> 'a list
    let runList b t = 
        let (Behavior bf) = b
                 in List.map bf t


    // val runSeq : 'a Behavior -> seq<Time> -> seq<'a>
    let runSeq b t = let (Behavior bf) = b
                         in Seq.map bf t

    //runSeq time5B ([for i in [0. .. 10.] -> (float(i))] |> List.toSeq)

    module FRPState =

        type Time = float
        type 'a Behavior = Behavior of (Time -> ('a * 'a Behavior))
        type 'a Event =  Event of (Time -> ('a option  * 'a  Event))
    
        let rec runList (Behavior bf) times =
           match times with
           |[] -> []
           |h::t -> let (r, nb) = bf h
                    r :: runList nb t

        let rec doubleB =
            let bf t = (2.0 * t, doubleB)
            Behavior bf

        // Time Behavior
        let rec timeBeh = Behavior (fun t -> (t, timeBeh))


        // 'a -> 'a Behavior
        let constBeh v =
            let rec bf t = (v, Behavior bf)
            Behavior bf

        // int Behavior
        let rec fiveBeh =
           let bf t = (5, fiveBeh)
           Behavior bf

        let beh a = (fun x -> a * Behavior)
        
        // (Time -> 'a) -> 'a Behavior
        let createBehavior f =
            let rec bf t = (f t, Behavior bf)
            Behavior bf
        
        // val runList : 'a Behavior -> Time list -> 'a list
        let rec runList (Behavior bf) times =
           match times with
           |[] -> []
           |h::t -> let (r, nb) = bf h
                    r :: runList nb t

        // Time list
        let timeList = Seq.toList (seq { for x in 1 .. 25 -> (float) x})

        // Time -> Time -> float * time
        let f t0 t = if (t0-t < 10.0) then (t0-t, t0) else (0.0, t)

        // (Time -> float * Time) list
        runList (createBehavior f) timeList |> List.iter (fun x -> printf "%A" (x(1.)))


        // 'a Behavior -> (Time list -> 'a list)
        let rec runBehavior (Behavior bf) = function
            |[] -> []
            |h::t -> let (r, nb) = bf h // h = Time
                     r::(runBehavior nb t)
        
        type Color = Green | Red

        // a constant Behavior that is always Red
        // Color Behavior
        let rec redBehavior = Behavior (fun t -> (Red, redBehavior))

        // Color Behavior
        let rec colorBehavior =
           let bf t =  if (t < 10.0)
                       then (Green, colorBehavior)
                       else (Red, redBehavior)
           Behavior bf


        

        let r = runBehavior (*'a Behavior -> (Time list -> 'a list)*) colorBehavior timeList

        // (Time -> Color Behavior option) -> Color Behavior
        let rec colorBehavior cond =
           // Time -> Color * Color Behavior
           let bf t = match cond t with
                      |None -> (Green, colorBehavior cond)
                      |Some (Behavior newColorB) -> newColorB t
           Behavior bf

        // val cond : float -> Color Beh option
        let cond t = if (t<10.0)
                     then None
                     else Some redBehavior

//        runBehavior (*'a Behavior -> (Time list -> 'a list)*) 
//                (colorBehavior cond) timeList
//


        //let r = runBehavior  colorB s

        let rec condBehavior =
           let bf t = if (t<10.0) then (None, condBehavior)
                      else (Some redBehavior, condBehavior)
           Behavior bf

        // val switchB : 'a Beh -> 'a Beh Event -> 'a Beh

        let rec switchB (Behavior bfInit) (Event condf) =
           let bf t = match condf t with
                      |(None, ncond) -> let (rInit, nBInit) = bfInit t
                                        (rInit, switchB nBInit ncond)
                      |(Some (Behavior newB), ncond) -> newB t
           Behavior bf
//
        let rec condB =
           let bf t = if (t<10.0)
                      then (None, condB)
                      else let rec blackB = Behavior (fun t -> (Red, blackB))
                           (Some blackB, condB)
           Behavior bf





        // val switch : 'a Behavior -> 'a Behavior option Behavior -> 'a Behavior
        let rec switch (Behavior initial) (Behavior cond) =
           // Time -> 'a * 'a Behavior
           let bf t = match cond t with
                      |(None, ncond) -> let (rInit, nBInit) = initial t
                                        (rInit, switch nBInit ncond)
                      |(Some (Behavior newBehavior), ncond) -> newBehavior t
           Behavior bf

        let rec redBehavior = Behavior (fun t -> (Red, redBehavior))

        let colorBehavior = switch redBehavior condB

        runBehavior colorBehavior timeList


        // val switch : 'a Behavior -> 'a Behavior Event -> 'a Behavior
        let rec switch (Behavior bfInit) (Event condf) =
            let bf t = match condf t with
                        |(None, ncond) -> let (rInit, nBInit) = bfInit t
                                          (rInit, switch nBInit ncond)
                        |(Some (Behavior newB), ncond) -> newB t
            Behavior bf