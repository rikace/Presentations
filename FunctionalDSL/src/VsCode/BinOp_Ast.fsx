
open System
open System.Windows.Forms

[<STAThread>]
let _ =
    let form = new Form(Text="Example", Width=400, Height=300)
    new Button(Text="Click me...") |> form.Controls.Add
    new Button(Text="And me...", Left=80) |> form.Controls.Add
    form |> Application.Run
    
    
type BOp = BPlus | BMinus | BTimes | BDivide

type Constant = 
    | FConst of float 
    | IConst of int

type Expr = BinOp of BOp * Expr * Expr
          | Const of Constant

let inline binEval op e1 e2 = 
    match op with 
    | BPlus -> e1 + e2
    | BMinus -> e1 - e2
    | BTimes -> e1 * e2
    | BDivide -> e1 / e2

let inline (.+) e1 e2 = BinOp (BPlus, e1, e2)
let F v = Const (FConst v)
let I v = Const (IConst v)



let rec eval e =
  match e with
  | BinOp (op, e1, e2) -> binEval op (eval e1) (eval e2)
  | Const c -> match c with
               | FConst f -> f
               | IConst i -> float i
 
let result = BinOp (BPlus, Const (IConst 7), (BinOp (BTimes, Const (IConst 10), Const (IConst 2))))

result |> eval

let x = I 7 .+ I 5 |> eval

//

accounts 
|> Seq.filter (belongsTo "John S.")
|> Seq.map calculateInterest
|> Seq.filter (flip (>) threshold)
|> Seq.fold (+) 0.0



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