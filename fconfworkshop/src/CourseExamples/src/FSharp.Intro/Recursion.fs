module Recursion

let rec factorial n =
    if n = 0 then
        1
    else n * factorial (n - 1)

let tailRecFactorial n =
    let rec fact n acc =
        if n = 0 then
            acc
        else
            fact (n - 1) acc * n
    fact n 1