namespace Easj360FSharp 

open System

module OptionalPattern = 

    let (|Even|Odd|) n = 
        match n % 2 with
        | 0 -> Even
        | _ -> Odd


    let isEven n =
        match n with 
        | Even n -> true
        | Odd n -> false



