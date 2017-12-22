type Expression =
    | X
    | Constant of float
    | Add of Expression * Expression
    | Mul of Expression * Expression

let rec interpret (ex:Expression) =
    match ex with
    | X -> fun (x:float) -> x
    | Constant(value) -> fun (x:float) -> value
    | Add(leftExpression,rightExpression) ->
        let left = interpret leftExpression
        let right = interpret rightExpression
        fun (x:float) -> left x + right x
    | Mul(leftExpression,rightExpression) ->
        let left = interpret leftExpression
        let right = interpret rightExpression
        fun (x:float) -> left x * right x

let run (x:float,expression:Expression) =
        let f = interpret expression
        let result = f x
        printfn "Result: %.2f" result


let expression = Add(Constant(1.0),Mul(Constant(2.0),X))
run(10.0,expression)

let expression2 = Mul(X,Constant(10.0))
run(10.0,expression2)

let add a b = Add(a, b)
let mul a b = Mul(a, b)
let c v =  Constant v
let x = X

let expression3 = add (c 1.0) (mul (c 2.0) x)
run(10.0,expression3)



(*
>
Result: 21.00
Result: 100.00
*)