module AttemptBuilder 
    
type Attempt<'a> = (unit -> 'a option)

let succeed x = (fun () -> Some(x))
let fail = (fun () -> None)
let runAttempt (a: Attempt<'a>) = a()
let delay f =(fun () -> runAttempt (f ()))

type Attempt = 
    static member Fail() = fail
    static member Succeed x = succeed x
    static member Run (a: Attempt<'a>) = runAttempt a

let rec bind p rest i max =
    try
        match runAttempt p with
        | Some r -> (rest r)
        | None -> fail
    with
    |_ when i < max -> bind p rest (i + 1) max
    |_ -> fail


type AttemptBuilder(maxRetry:int) =
    member b.Return x = succeed x
    member b.ReturnFrom (x: Attempt<'a>) = x
    member b.Bind(p, rest) = bind p rest 1 maxRetry
    member b.Delay f = delay f
    member b.Zero() = fail


let attempt x = AttemptBuilder(x)
let attemptOne = attempt 1

let test : Attempt<int> = attemptOne { printfn "Running and throwing error"; failwith "oooops" }
let testn n : Attempt<int> = attempt n { printfn "Running and throwing error"; failwith "oooops" }

let run1 = Attempt.Run <| attemptOne {  let! r = test
                                        return r }

let run3 = Attempt.Run <| attempt 3 {  let! r = test
                                       return r }

let runOk = Attempt.Run <| attempt 3 { return 42 }

let test2 n : Attempt<int> = attemptOne { printfn "Ok from test2"
                                          if n > 100 then return n
                                          else failwith "oooops" }

let failIf n = Attempt.Run <| attempt 2 { let! res = test2 n 
                                          printfn "Ok from failIf"
                                          return res }
failIf 101                                              

 



