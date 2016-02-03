[<EntryPoint>]
let main argv = 

    let xs = [1;3;5;7;9;11]
    let ys = [4;5;6;7;11;13]
    
    // 'a list -> a:'a -> 'a option
    let inList lst a = List.tryFind(fun f -> f = a) lst
    List.map(inList xs) ys // [None; Some 5; None; Some 7; Some 11; None]

    // int list -> int list -> bool list
    let rec inList' (xs:int list) (ys:int list) =
        match xs, ys with
        | [], ys -> ys |> List.map(fun _ -> false)
        | xs, [] -> []
        | x::tx, y::ty ->   if x < y then inList' tx ys
                            else (x = y)::(inList' xs ty)
    inList' xs ys // [false; true; false; true; true; false]


    printfn "%A" argv
    0 // return an integer exit code
