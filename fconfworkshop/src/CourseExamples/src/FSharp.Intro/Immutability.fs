module Immutability

let value = 1
printfn "%d" value
value = 2
printfn "%d" value
printfn "%b" (value = 2)

let mutable value2 = 1
printfn "%d" value2
value2 <- 2
printfn "%d" value2