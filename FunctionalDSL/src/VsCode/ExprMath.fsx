
// =====
(*
    Now that we have the building blocks to represent ideas 
    in F#, we have all the power we need to represent a real world problem in the language of mathematics.

    In this simple example we were able to represent and evaluate a four-function mathematical expression using only a discriminated union and a pattern match. You would be hard 
    pressed to write the equivalent C# in as few lines of code because you would need to add additional scaffolding to represent these concepts.
*)
// This Discriminated Union is sufficient to express any four-function
// mathematical expression.
type Expr =
    | Num      of int
    | Add      of Expr * Expr
    | Subtract of Expr * Expr
    | Multiply of Expr * Expr
    | Divide   of Expr * Expr
    
// This simple pattern match is all we need to evaluate those
// expressions. 
let rec evaluate expr =
    match expr with
    | Num(x)             -> x
    | Add(lhs, rhs)      -> (evaluate lhs) + (evaluate rhs)
    | Subtract(lhs, rhs) -> (evaluate lhs) - (evaluate rhs)
    | Multiply(lhs, rhs) -> (evaluate lhs) * (evaluate rhs)
    | Divide(lhs, rhs)   -> (evaluate lhs) / (evaluate rhs)

// 10 + 5
let ``10 + 5`` = 0

// 10 * 10 - 25 / 5
let sampleExpr = 
    Subtract(
        Multiply(
            Num(10), 
            Num(10)),
        Divide(
            Num(25), 
            Num(5)))
        
let result = evaluate sampleExpr


// It appears that building an internal LOGO-like DSL is surprisingly easy, and requires almost no code! What you need is just to define the basic types to describe your actions:




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