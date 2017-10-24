open DataModel
open System

open System.Data.Linq.SqlClient
open System.Linq
open System.Linq.Expressions
open FSharp.Linq.NullableOperators
open Microsoft.FSharp.Quotations
open LinqKit
open System.ComponentModel.DataAnnotations
open System.Data.Entity
open Microsoft.FSharp.Quotations.Patterns


[<AutoOpen>]
module mystuff =
    type Operator =
      | Plus
      | Minus
//      | Divide
//      | Multiply

    type RelationOperator =
        | Equal
        | NotEqual
        | LessThan
        | LessThanOrEqual
        | GreaterThan
        | GreaterThanOrEqual


    type Expr =
      | Property of Type * string
      | Operation of Expr * Expr * Operator
      | Binary of Expr * Expr * RelationOperator
      | Constant of Value

    and Value =
      | String of string
      | Decimal of decimal
      | Bool of bool
      | SByte of sbyte
      | Int32 of int32
      | Int16 of int16
      | Single of single
      | Double of double

    type Predicate =
      | Condition of Expr * Expr * RelationOperator
      | And of Predicate * Predicate
      | Or of Predicate * Predicate
      | Not of Predicate


    type ExpressionFilter =
        | GetProperty of Expr * string
        | Filter of Predicate


    type 'a evalExpr = ExpressionFilter -> 'a



    type Aggregation =
     | GroupKey
     | CountAll
     | CountDistinct of string
     | ReturnUnique of string
     | ConcatValues of string
     | Sum of string
     | Mean of string

    type SortDirection =
      | Ascending
      | Descending

    type Paging =
      | Take
      | Skip

    type Transformation =
      | Columns of string list
      | SortBy of (string * SortDirection) list
      | GroupBy of string list * Aggregation list
      | Paging of string * Paging list
      | GetSeries of string * string
      | Empty




    let splitAtChar (c : char) (s : string) : string array =
            s.Split([| c |], System.StringSplitOptions.RemoveEmptyEntries)

    let property (propertyName : string) (param : ParameterExpression) =
        propertyName
            |> splitAtChar '.'
            |> Seq.fold (fun state property -> Expression.Property(state, property) :> Expression) (param :> Expression)

    let parameter<'entity> = Expression.Parameter(typeof<'entity>, "i")

    let constant<'a> (constant : 'a) = Expression.Constant(constant, typeof<'a>)
    let convert<'a> expression = Expression.Convert(expression, typeof<'a>)
    let invoke expression args = Expression.Invoke(expression, args)
    let or' left right = Expression.Or(left, right)
    let ifThenElse test left right = Expression.IfThenElse(test, left, right)

    let lambda<'entity, 'a> body parameters = Expression.Lambda<Func<'entity, 'a>>(body, parameters).Expand()

    let asExpr x = (x :> Expression)

    let mapRelationOperator (op:RelationOperator) exp1 exp2 =
        match op with
        | Equal -> Expression.Equal(exp1,exp2)
        | NotEqual -> Expression.NotEqual(exp1,exp2)
        | LessThan -> Expression.LessThan(exp1,exp2)
        | LessThanOrEqual -> Expression.LessThanOrEqual(exp1,exp2)
        | GreaterThan-> Expression.GreaterThan(exp1,exp2)
        | GreaterThanOrEqual-> Expression.GreaterThanOrEqual(exp1,exp2)



    let binary relation exp1 exp2 =
        (exp1, exp2)
        |> match relation with
            | RelationOperator.Equal -> Expression.Equal
            | RelationOperator.NotEqual -> Expression.NotEqual
            | RelationOperator.LessThan -> Expression.LessThan
            | RelationOperator.LessThanOrEqual -> Expression.LessThanOrEqual
            | RelationOperator.GreaterThan -> Expression.GreaterThan
            | RelationOperator.GreaterThanOrEqual -> Expression.GreaterThanOrEqual

    let comparison<'entity, 'a> (propertyName : string) relation (cons : 'a) =
        let param = parameter<'entity>
        let propExpr = param |> property propertyName
        let constExpr = constant cons
        let body = binary relation propExpr constExpr
        lambda<'entity, bool> body [| param |]


    let rec evalExpr exp =
        match exp with
        | Operation (expr1, expr2, op) ->
                printfn "Oper"
                match op with
                | Plus -> Expression.Add((evalExpr expr1), (evalExpr expr2))  |> asExpr
                | Minus -> Expression.Subtract((evalExpr expr1), (evalExpr expr2)) |> asExpr
        | Property (ty,s) ->
                printfn "Property %s - type %s" s (ty.Name)
                let param = Expression.Parameter(ty, "i")
                param |> property s
        | Binary(exp1, exp2, op) ->
                printfn "Binary %A" op
                mapRelationOperator op (evalExpr exp1) (evalExpr exp2) |> asExpr
        | Constant value ->
                printfn "Const %A" value
                let o, ty =
                    match value with
                    | String v -> (box v, typeof<string>)
                    | Decimal v -> (box v, typeof<decimal>)
                    | Bool v -> (box v, typeof<bool>)
                    | SByte v ->(box v, typeof<sbyte>)
                    | Int32 v -> (box v, typeof<int32>)
                    | Int16 v -> (box v, typeof<int16>)
                    | Single v -> (box v, typeof<single>)
                    | Double v ->  (box v, typeof<double>)
                Expression.Constant(o,ty) |> asExpr


    let rec eval (p:Predicate) = //: Linq.Expressions.Expression =
        match p with
        | Condition(Property(ty,s), exp2, rel) ->
                printfn "Condition"
                //(evalExpr (Binary(exp1, exp2, rel))) |> asExpr
                let param = Expression.Parameter(ty, "i")

              //  let y = (evalExpr exp1) :?> MemberExpression
                let body = (binary rel ( param |> property s) (evalExpr exp2))
                Expression.Lambda(body, param) |> asExpr
        | Condition(exp1, exp2, rel) -> (binary rel (evalExpr exp1) (evalExpr exp2)) |> asExpr

        | And(p1, p2) ->
            printfn "And"
            Expression.And((eval p1) , (eval p2))|> asExpr
        | Or(p1, p2) ->
            printfn "Or"
            Expression.Or((eval p1) , (eval p2)) |> asExpr
        | Not p ->
            printfn "Not"
            Expression.Not(eval p) |> asExpr
//
    let student<'t> id =
            Condition(Property(typeof<'t>, "Age"), Constant(Int32(id)), RelationOperator.GreaterThanOrEqual)



[<EntryPoint>]
let main argv =

    let testCom = comparison<Student, int> "Age" RelationOperator.GreaterThanOrEqual 1

    let w  = (student<Student> 4 |> eval).Expand()

   /// let lambda<'entity, 'a> body parameters = Expression.Lambda<Func<'entity, 'a>>(body, parameters)
  //  let lam : Expression<Func<Student,bool>> = lambda<Student, bool> (w.Expand())




    use ctx = new SimpleDbContext("Data Source=DESKTOP-195V044;Initial Catalog=SimpleDB;Integrated Security=SSPI;")
    //let students = ctx.Students.ToArray()
    let students = ctx.Set<Student>().AsQueryable() //.Where(lam.Compile())

    // predicates
    // eval/interpret predicates
    // exec ctx.Students.Where(....)


    for student in students do
        printfn "Name %s, %s - Age %A" student.FirstName student.LastName student.Age

    Console.ReadLine() |> ignore
    0
