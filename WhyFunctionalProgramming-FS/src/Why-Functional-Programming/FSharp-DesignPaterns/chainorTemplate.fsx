
// chain template function
let chainTemplate processFunction canContinue s = 
    if canContinue s then 
        processFunction s
    else s

let canContinueF _ = true
let processF x = x + 1

//combine two functions to get a chainFunction
let chainFunction = chainTemplate processF canContinueF   

// use pipeline to form a chain
let s = 1 |> chainFunction |> chainFunction

printfn "%A" s