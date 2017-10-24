module MorseCodeInterpreter

open System
open System.Reactive.Linq
open MorseCode

let splitBy (separator : 'a) (source : IObservable<'a>) =
    source
        .Window(fun () -> source.Where(fun x -> x = separator))
        .Select(fun ys -> ys.Where(fun y -> y <>  separator))

let processChar acc ch =
    match acc with
    | Node(_, dash, dot) ->
        match ch with
        | '.' ->  dot
        | '-' -> dash
        | _ -> acc
    | _ -> acc

let translate (source : IObservable<char>) =
    (source
     |> splitBy ' ')
        .Select(fun xs -> xs
                            .Scan(startNode, fun acc ch -> processChar acc ch)
                            .Monitor("Morse Code Scan", 1.0)
                            .Select(fun x -> match x with
                                             | Node(value, _, _) -> value
                                             | Leaf(value) -> value
                                             | _ -> ""))