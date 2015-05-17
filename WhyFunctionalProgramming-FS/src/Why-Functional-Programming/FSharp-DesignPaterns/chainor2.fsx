//this version use the pipeline to chain responsibilities  

let chainTemplate processFunction canContinue s = 
    if canContinue s then 
        processFunction s
    else s

let canContinueF _ = true
let processF x = x + 1

let chainFunction = chainTemplate processF canContinueF   //combine two functions to get a chainFunction
let s = 1 |> chainFunction |> chainFunction

printfn "%A" s

