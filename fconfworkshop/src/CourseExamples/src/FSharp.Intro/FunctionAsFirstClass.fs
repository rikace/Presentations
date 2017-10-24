module FunctionAsFirstClass

let add a b = a + b

let map (f : 'a -> 'b) (xs : 'a seq) =
    seq { for x in xs do
              yield f x }

let newList = [1..10] 
              |> map (fun x -> x + 1)

let add1 = add 1
let value = add1 2