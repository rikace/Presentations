module Computation


[<AutoOpen>]
module Computation =
    open System
    open Microsoft.FSharp.Math

    let maxIteration = 255
    let modSquared (c : Complex) = 
        c.RealPart * c.RealPart + c.ImaginaryPart * c.ImaginaryPart
    
    let (|Escaped|DidNotEscape|) c =
        let rec compute z iterations =
            if(modSquared z >= 4.0) 
                then Escaped iterations
            elif iterations = maxIteration
                then DidNotEscape
            else compute ((z * z) + c) (iterations + 1)
        compute c 0