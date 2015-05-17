module Recursion

type Tree<'a> =
| Branch of 'a *Tree<'a> * Tree<'a>
| Empty 

with member this.WalkTree f =
            let rec walkTree tree =
                    match tree with
                    | Branch(x, left, right) -> 
                            do f(x)
                            walkTree left
                            walkTree right
                    | _ -> ()
            walkTree this

let tree = Branch(4, Branch(7, Branch(3, Empty,Empty), Branch(9,Empty,Empty)), Empty)
tree.WalkTree(printfn "Node Value %d")


let sumEvenNumberUntill200 (ints:int list) =
    let rec sumEvenNumberUntill200 acc lst =
            match lst with
            | [] -> acc
            | head::tail when head % 2 = 0 -> 
                    if acc + head > 200 then acc
                    else sumEvenNumberUntill200 (acc + head) tail
            | _::tail -> sumEvenNumberUntill200 acc tail
    sumEvenNumberUntill200 0 ints
        

let ``182`` = sumEvenNumberUntill200 [1..100] // 992
let ``90`` = sumEvenNumberUntill200 [1..18]


let rec badFact n =
    if n = 0 then 1
    else n * (badFact (n - 1))

badFact 1000000


let goodFact n =
    let rec fact i acc =
        if i = 0 then acc
        else fact (n - 1) (n * acc)
    fact n 1

goodFact 100