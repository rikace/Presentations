module ExpressionUtil 

open DataModel
open System.ComponentModel.DataAnnotations
open System.Data.Entity
open System
open System.Reflection
open System.Linq.Expressions
open System.Linq

open Microsoft.FSharp.Quotations
open Microsoft.FSharp.Quotations.Patterns

let inline toExpression (expr : Expr) : Expression =
    Linq.RuntimeHelpers.LeafExpressionConverter.QuotationToExpression(expr)

let inline evalExpr<'T> (expr : Expr) =
    Linq.RuntimeHelpers.LeafExpressionConverter.EvaluateQuotation(expr) :?> 'T

let inline toLambdaExpression (expr : Expr<'a>) : Expression<'a> =
    Linq.RuntimeHelpers.LeafExpressionConverter.QuotationToLambdaExpression(expr)

let inline toExpressionTy (expr : Expr<'a>) : Expression<'a> =
    Linq.RuntimeHelpers.LeafExpressionConverter.QuotationToExpression
    |> unbox<Expression<'a>>

let ``Exp _ -> true``<'a> = <@ Func<_,bool>(fun _ -> true) @> |>  Linq.RuntimeHelpers.LeafExpressionConverter.QuotationToLambdaExpression

let rec exprLambda (expr:Expr<'a -> 'b>) =
        match expr with
        | Patterns.Lambda(var, body) ->
            let bodyExpr = convertExpr body
            let param = Expression.Parameter(var.Type, var.Name)
            Expression.Lambda<Func<'a, 'b>>(param)
        | _ -> failwith "not supported"
and private convertExpr expr =
    match expr with
    | Patterns.Var(var) ->
        Expression.Variable(var.Type, var.Name) :> Expression
    | Patterns.PropertyGet(Some inst, pi, []) ->
        let instExpr = convertExpr inst
        Expression.Property(instExpr, pi) :> Expression
    | Patterns.Call(Some inst, mi, args) ->
        let argsExpr = Seq.map convertExpr args
        let instExpr = convertExpr inst
        Expression.Call(instExpr, mi, argsExpr) :> Expression
    | Patterns.Call(None, mi, args) ->
        let argsExpr = Seq.map convertExpr args
        Expression.Call(mi, argsExpr) :> Expression
    | _ -> failwith "not supported"

// ex contains <@ fun (p:IPlayer) -> p.id @> 
let containsExpr (membr: Expr<'a -> 'b>) (vals: 'b list) = 
    match membr with
    | Lambda (_, PropertyGet _) -> 
        match vals with
        | [] -> <@ fun _ -> true @>
        | _ -> vals |> Seq.map (fun v -> <@ fun a -> (%membr) a = v @>) |> Seq.reduce (fun a b -> <@ fun x -> (%a) x || (%b) x @>)
    | _ -> failwith "Expression has to be a member"

let containsToExpr (membr: Expr<'a -> 'b>) (vals: 'b list) =  (vals |> containsExpr membr) |> toExpressionTy 
