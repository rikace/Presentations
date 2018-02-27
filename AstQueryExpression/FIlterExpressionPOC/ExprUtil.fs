module ExprUtil


    open System
    open System.Linq
    open System.Linq.Expressions
    open System.Collections.Generic
    open System.Threading
    open System.Threading.Tasks
    open FSharp.Quotations
    open FSharp.Quotations.Patterns



    [<RequireQualifiedAccess>]
    module String =
        let containsIgnoreCase sought (source : string) = source.IndexOf(sought, StringComparison.OrdinalIgnoreCase) >= 0
        let [<Literal>] empty = ""
        let splitAtChar (c : char) (s : string) : string array =
            s.Split([| c |], System.StringSplitOptions.RemoveEmptyEntries)



    type Relation  =
        | Equal
        | NotEqual
        | LessThan
        | LessThanOrEqual
        | GreaterThan
        | GreaterThanOrEqual


    [<CompilationRepresentation(CompilationRepresentationFlags.ModuleSuffix)>]
    module Type =
        let name o = o.GetType().Name

        let isAssignableFrom (actual:Type) (expected:Type) =
            expected.IsInterface && expected.IsAssignableFrom actual

    type ExpressionValidationResult = Valid | Invalid of string | WrongType

    let requireType<'actual, 'expected> fn =
        match typeof<'actual> with
        | t when t = typeof<'expected> -> fn()
        | t when typeof<'expected> |> Type.isAssignableFrom t -> fn()
        | _ -> ExpressionValidationResult.WrongType

    let requireTypeOnly<'actual, 'expected> = requireType<'actual, 'expected> (fun () -> ExpressionValidationResult.Valid)

    let nameof<'entity, 'a> (property : Expr<'entity -> 'a>) =
        let state = String.empty
        let rec propertyNameGet state quotation =
            match quotation with
            | PropertyGet (_,p,_) ->
                match state with
                | String.empty -> p.Name
                | _ -> state + "." + p.Name
            | Lambda (_,expr) -> propertyNameGet state expr
            | Let (_,e1,e2) -> propertyNameGet (propertyNameGet state e1) e2
            | _ -> failwith "Property name cannot be derived from the quotation passed to propName"

        propertyNameGet state property

    let parameter<'entity> = Expression.Parameter(typeof<'entity>, "i")
    let constant<'a> (constant : 'a) = Expression.Constant(constant, typeof<'a>)
    let convert<'a> expression = Expression.Convert(expression, typeof<'a>)
    let invoke expression args = Expression.Invoke(expression, args)
    let or' left right = Expression.Or(left, right)
    let ifThenElse test left right = Expression.IfThenElse(test, left, right)
    let lambda<'entity, 'a> body parameters = Expression.Lambda<Func<'entity, 'a>>(body, parameters) // .Expand()

    let property (propertyName : string) (param : ParameterExpression) =
        // Builds up a MemberExpression that navigates to
        // nested properties
        propertyName
            |> String.splitAtChar '.'
            |> Seq.fold (fun state property -> Expression.Property(state, property) :> Expression) (param :> Expression)

    let binary relation exp1 exp2 =
        (exp1, exp2)
        |> match relation with
            | Relation.Equal -> Expression.Equal
            | Relation.NotEqual -> Expression.NotEqual
            | Relation.LessThan -> Expression.LessThan
            | Relation.LessThanOrEqual -> Expression.LessThanOrEqual
            | Relation.GreaterThan -> Expression.GreaterThan
            | Relation.GreaterThanOrEqual -> Expression.GreaterThanOrEqual


    let accessor<'entity, 'a> (propertyName : string) =
        let param = parameter<'entity>
        let body = param |> property propertyName
        lambda<'entity, 'a> body [| param |]

    let comparison<'entity, 'a> (propertyName : string) relation (cons : 'a) =
        let param = parameter<'entity>
        let propExpr = param |> property propertyName
        let constExpr = constant cons
        let body = binary relation propExpr constExpr
        lambda<'entity, bool> body [| param |]

    let contains<'entity, 'a> (propertyName : string) (vals : 'a seq) =
        let param = parameter<'entity>
        let propExpr = param |> property propertyName
        let constExpr = constant vals
        let body = vals.GetType().GetMethod("Contains")
        let call = Expression.Call(constExpr, body , propExpr)
        lambda<'entity, bool> call [|param|]

    let notContains<'entity, 'a> (propertyName : string) (vals : 'a seq) =
        let param = parameter<'entity>
        let propExpr = param |> property propertyName
        let constExpr = constant vals
        let body = vals.GetType().GetMethod("Contains")
        let call = Expression.Call(constExpr, body , propExpr)
        let notContainsCall = Expression.Not(call)
        lambda<'entity, bool> notContainsCall [|param|]

    let boolean<'entity> (propertyName : string) b = comparison<'entity, bool> propertyName Relation.Equal b
    let isTrue<'entity> (propertyName : string) = boolean<'entity> propertyName true
    let isFalse<'entity> (propertyName : string) = boolean<'entity> propertyName false




    // when ( type 't = type 'tin ) or ( typeof<'tin> |> Type.isAssignableFrom 't)
    [<AbstractClass>]
    type FilterExp<'tin when 'tin : not struct>( input : string ) =

        member this.input = input
        abstract member validate<'t> : unit -> ExpressionValidationResult
        default this.validate<'t> () = requireTypeOnly<'t, 'tin>

        abstract member where<'entity> : unit -> Expression<Func<'entity, bool>>

//
//
//
//    let personId id =
//            Condition(Property("person_id"), Constant(Int32(id)), RelationOperator.Equal)
//
//    let teamID id = Condition(Property("team_id"), Constant(Int32(id)), RelationOperator.Equal)
//
//
//    let teamAndPerson teamID personID =
//         And ((personId 41),  (teamID 7))
//
//
//    let rec eval (p:Predicate) exp =
//        match p with
//        | Condition(exp1, exp2, rel) ->
//        | And(p1, p2) -> Expression.And((eval p1), (eval p2))
//        | Or(p1, p2) -> ()
//        | Not p -> ()




