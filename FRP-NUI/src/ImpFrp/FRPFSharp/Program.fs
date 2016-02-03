
open System

type Time = float
// Behaviors (signals) are flows of values, punctuated by event occurrences.
type 'a Behavior = Beh of (Time -> 'a)
type 'a Event = Evt of (Time * 'a) list // monotonic




[<EntryPoint>]
let main argv = 

//    let pure f = Beh f
//    let (<*>) f a = Beh (f a)
//

    0