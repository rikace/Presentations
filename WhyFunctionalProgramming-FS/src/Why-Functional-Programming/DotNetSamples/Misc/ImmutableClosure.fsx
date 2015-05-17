module immutableClosure =
    open System

    let actions = List.init 5 (fun i -> fun() -> i * 2)

    for action in actions do
        printfn "%d " (action())