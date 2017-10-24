module PredicateModel

open DataModel
open System.ComponentModel.DataAnnotations
open System.Data.Entity
open System
open System.Reflection
open System.Linq.Expressions
open System.Linq
open System.Reflection
open FSharp.Quotations
open FSharp.Quotations.Patterns
open System.ComponentModel
open Microsoft.FSharp.Reflection
open Microsoft.FSharp.Quotations

type Aggregation =
  | GroupKey
  | CountAll
  | CountDistinct of string
  | ReturnUnique of string
  | ConcatValues of string
  | Sum of string
  | Mean of string

type Paging =
  | Take of int
  | Skip of int

type SortDirection =
    | Desc
    | Asc

type Transformation =
  | SortBy of (string * SortDirection) list
  | GroupBy of string list * Aggregation list
  | FilterBy of (string * bool * string) list
  | Paging of Paging list


module QueryModel =

    type Column = string

    // Constants that can appear in expressions using `QueryExpression.Constant`
    type QueryValue =
      | Number of int
      | String of string
      | Boolean of bool
      | Decimal of decimal
      | SByte of sbyte
      | Single of single
      | Double of double
      | NullNumber of Nullable<int>
      | NullBoolean of Nullable<bool>
      | NullDecimal of Nullable<decimal>
      | NullSByte of Nullable<sbyte>
      | NullSingle of Nullable<single>
      | NullDouble of Nullable<double>

    let inline ZeroVal<'a when 'a : struct and 'a : (new : unit ->  'a) and 'a :> ValueType>() = System.Nullable<'a>()
    let inline Val<'a when 'a : struct and 'a : (new : unit ->  'a) and 'a :> ValueType>() (v:'a)= System.Nullable<'a>(v)


    // Numerical operators (just + and -). Take two numerical
    // expressions and produce new numerical expression.
    type NumericalOperator =
      | Plus
      | Minus
      | Divide
      | Mul

    // Relational operators (just > and <). Note that `=` is included in `QueryExpression`
    // as a separate case. This is because `<` and `>` can be applied only on numbers, but
    // `=` can be used to test equality of anything (numbers, but also strings)
    type RelationOperator =
      | LessThan
      | GreaterThan
      | LessThanOrEqual
      | GreaterThanOrEqual

    // Assuming the model supports a small fixed number of other methods, we list them
    // in this DU - for now we can check string length & string contains
    // (By representing the methods as DU cases and not, e.g. quotation, we can easily
    // pattern match on them)
    // it might make sense to just store their `MethodInfo`
    type KnownMethod =   // TODO more here as it goes
      | StringContains
      | StringLength
      | StringStartsWith
      | StringEndsWith
      | SeqContainInt
      | SeqContainNullInt


      // Generate `MethodGetter` for mapping method names to MethodInfo
      // (this is needed for `ConversionContext`)
    let geSeqMethod ty =
            typeof<Enumerable>.GetMethods()
            |> Seq.filter(fun m -> m.Name = "Contains")
            |> Seq.find(fun m -> m.GetParameters().Length = 2)
            |> fun m -> m.MakeGenericMethod([|ty|])

    let getKnownMethod this =
            match this with
            | StringLength -> typeof<string>.GetProperty("Length").GetGetMethod()
            | StringContains -> typeof<string>.GetMethod("Contains")
            | StringStartsWith -> typeof<string>.GetMethod("StartsWith")
            | StringEndsWith -> typeof<string>.GetMethod("EndsWith")
            | SeqContainInt ->
                            printfn "Call SeqContainInt"
                            geSeqMethod typeof<int>
            | SeqContainNullInt ->
                            printfn "Call SeqContainNULL"
                            geSeqMethod typeof<int Nullable>


    type CollectionReference =
      | InMemoryInts of SeqInts
      | InMemoryNullInts of SeqNullInts
    and SeqNullInts = System.Collections.Generic.IEnumerable<System.Nullable<int>>
    and SeqInts = System.Collections.Generic.IEnumerable<int>

    // Represents expression, which can be evaluated to value of various types
    type QueryExpression =
      | GetColumn of string
      | NumericalBinary of QueryExpression * QueryExpression * NumericalOperator
      | RelationalBinary of QueryExpression * QueryExpression * RelationOperator
      | Equals of QueryExpression * QueryExpression
      | Call of KnownMethod * QueryExpression list
      | Constant of QueryValue
      | ContainsElements of CollectionReference * QueryExpression



    // Represents a predicate, which is always a boolean value
    type QueryPredicate =
      | Condition of QueryExpression
      | And of QueryPredicate * QueryPredicate
      | Or of QueryPredicate * QueryPredicate
      | Not of QueryPredicate

    // Sorting
    type ColumnInfo = {ColumnName:string;ColumnType:Type}
    and SortDirection =
        | Ascending
        | Descending
    and SortBy = SortBy of (ColumnInfo * SortDirection) list

    // ------------------------------------------------------------------------------------------------
    // Translation - The code takes our `QueryPredicate` and generates
    // LINQ expression tree (a value of type `Expression<Func<'T, bool>>` which we can then pass
    // to the LINQ `Where` method on `DbSet<'T>`. LINQ translates the expression tree to SQL and
    // runs it. The translation uses expression trees directly, but there are some alternatives:
    //
    //  * Instead of generating LINQ `Expression`, we could generate F# quotations. Quotations could
    //    then be translated to LINQ expressions using `LeafExpressionConverter.QuotationToLambdaExpression`
    //    from `Microsoft.FSharp.Linq.RuntimeHelpers`. We could then use quotations like <@@ %%e1 + %%e2 @@>
    //    in some places of the translation rather than `Expression.Add`. This would be a bit shorter.
    //
    // There is also a function that extracts all column names that are used in the `QueryPredicate`.
    // (this could be used to select the appropriate table for before running the query)
    // ------------------------------------------------------------------------------------------------


    // When converting, the caller needs to tell us how to convert property access
    // and how to implement methods. `MethodGetter` returns whether the method is
    // instance method, number of parameters (including instance parameter) and
    // the `MethodInfo` to be used in the expression tree.
    type ConversionContext =
        { ColumnGetter : string -> Expression
          MethodGetter : KnownMethod -> bool * int * MethodInfo }

[<AutoOpen>]
module UtilQuery =
    open QueryModel

    let columnInfo<'entity, 'a> (property : Expr<'entity -> 'a>) =
        let rec propertyNameGet property =
            match property with
            | PropertyGet (_,p,_) -> {ColumnName=p.Name;ColumnType=p.PropertyType}
            | Lambda (_,expr) -> propertyNameGet expr
            | _ -> failwith "Property name cannot be derived from the quotation passed to propName"
        propertyNameGet property



module PredicateQueryModule =
    open QueryModel

    // Transform `QueryExpression` into LINQ `Expression`.
    let rec convertExpression2 ctx e =
        match e with
        | Constant(Number n) -> <@@ n @@>
        | Constant(Boolean n) -> <@@ n @@>
        | NumericalBinary(e1, e2, NumericalOperator.Plus) -> <@@ %%(convertExpression2 ctx e1) + %%(convertExpression2 ctx e2) @@>
        | Call(KnownMethod.StringContains, [e1; e2]) -> <@@ (%%(convertExpression2 ctx e1):string).Contains(%%(convertExpression2 ctx e2)) @@>

    let equalWithNullValue (e1:Expression) (e2:Expression) =
        let hasValueExp = Expression.Property(e2, "HasValue")
        let valueExp = Expression.Property(e2, "Value")
        let equal = Expression.Equal(valueExp, e1)
        Expression.AndAlso(hasValueExp, equal)

    let equalbetweenNullValues (e1:Expression) (e2:Expression) =
        let hasValueExp1 = Expression.Property(e1, "HasValue")
        let valueExp1 = Expression.Property(e1, "Value")
        let hasValueExp2 = Expression.Property(e2, "HasValue")
        let valueExp2 = Expression.Property(e2, "Value")
        let equal = Expression.Equal(valueExp1, valueExp2)
        Expression.AndAlso(Expression.AndAlso(hasValueExp2, hasValueExp1), equal)


    let property (propertyName : string) (param : ParameterExpression) =
        // Builds up a MemberExpression that navigates to
        // nested properties
        propertyName.Split([|'.'|])
        |> Seq.fold (fun state property -> Expression.Property(state, property) :> Expression) (param :> Expression)

    // Transform `QueryExpression` into LINQ `Expression`.
    let rec convertExpression ctx e =
        match e with
        | GetColumn(s) -> ctx.ColumnGetter s
        | Constant(String v) -> upcast Expression.Constant(v)
        | Constant(Number v) -> upcast Expression.Constant(v)
        | Constant(Boolean v) -> upcast Expression.Constant(v)
        | Constant(Decimal v) -> upcast Expression.Constant(v)
        | Constant(SByte v) -> upcast Expression.Constant(v)
        | Constant(Single v) -> upcast Expression.Constant(v)
        | Constant(Double v) -> upcast Expression.Constant(v)
        | Constant(NullNumber(n)) -> upcast Expression.Constant(n, typeof<int Nullable>)
        | Constant(NullBoolean(n)) -> upcast Expression.Constant(n, typeof<bool Nullable>)
        | Constant(NullDecimal(n)) -> upcast Expression.Constant(n, typeof<decimal Nullable>)
        | Constant(NullSByte(n)) -> upcast Expression.Constant(n, typeof<sbyte Nullable>)
        | Constant(NullSingle(n)) -> upcast Expression.Constant(n, typeof<single Nullable>)
        | Constant(NullDouble(n)) -> upcast Expression.Constant(n, typeof<double Nullable>)
        | ContainsElements(coll, e) ->
            let e = convertExpression ctx e
            let mi, data =
                match coll with
                       | InMemoryInts(data) when e.Type = typeof<int Nullable> ->
                            let _, _, mi = ctx.MethodGetter SeqContainNullInt
                            mi, data.Select(fun n -> Nullable<int>(n)) |> Expression.Constant
                       | InMemoryNullInts(data) when e.Type = typeof<int> ->
                            let _, _, mi = ctx.MethodGetter SeqContainInt
                            mi, data.Where(fun n -> n.HasValue).Select(fun n -> n.Value) |> Expression.Constant
                       | InMemoryInts(data) ->
                            let _, _, mi = ctx.MethodGetter SeqContainInt
                            mi, data |> Expression.Constant
                       | InMemoryNullInts(data) ->
                            let _, _, mi = ctx.MethodGetter SeqContainNullInt
                            mi, data |> Expression.Constant
            upcast Expression.Call(mi, data, e)

        | Call(m, args) ->
            // Get information about the method, check that we
            // got the right number of arguments and generate call
            // `Expression.Call` has one overload for instance methods
            // and one overload for static methods.
            let isInstance, argCount, mi = ctx.MethodGetter m
            if List.length args <> argCount then
               failwith (sprintf "Error when converiting call to '%s'. Expected '%d' arguments but got '%d'" mi.Name argCount (List.length args))
            let args = List.map (convertExpression ctx) args
            if not isInstance then upcast Expression.Call(mi, args)
            else upcast Expression.Call(List.head args, mi, List.tail args)
//        | Equals((Constant(NullValue(_)) as e1), (Constant(NullValue(_)) as e2)) ->
//             let e1 = convertExpression ctx e1
//             let e2 = convertExpression ctx e2
//             upcast equalbetweenNullValues e2 e1
//        | Equals(e1, (Constant(NullValue(v)) as e2)) ->
//             let e1 = convertExpression ctx e1
//             let e2 = convertExpression ctx e2
//             upcast equalWithNullValue e1 e2
//        | Equals((Constant(NullValue(v)) as e1), e2) ->
//             let e1 = convertExpression ctx e1
//             let e2 = convertExpression ctx e2
//             upcast equalWithNullValue e2 e1
        | Equals(e1, e2) ->
            // Generate equality test - this should work for any
            // type, but LINQ's `Expression.Equal` seems to be clever
            // enough to handle that for us :-)
            let e1 = convertExpression ctx e1
            let e2 = convertExpression ctx e2
            upcast Expression.Equal(e1, e2)
        | RelationalBinary(e1, e2, op) ->
            let e1 = convertExpression ctx e1
            let e2 = convertExpression ctx e2
            match op with
            | LessThan -> upcast Expression.LessThan(e1, e2)
            | LessThanOrEqual -> upcast Expression.LessThanOrEqual(e1, e2)
            | GreaterThan -> upcast Expression.GreaterThan(e1, e2)
            | GreaterThanOrEqual -> upcast Expression.GreaterThanOrEqual(e1, e2)
        | NumericalBinary(e1, e2, op) ->
            let e1 = convertExpression ctx e1
            let e2 = convertExpression ctx e2
            match op with
            | Plus -> upcast Expression.Add(e1, e2)
            | Minus -> upcast Expression.Subtract(e1, e2)
            | Divide -> upcast Expression.Divide(e1, e2)
            | Mul -> upcast Expression.Multiply(e1, e2)

    // Transform `QueryPredicate` into LINQ `Expression`
    // (representing and expression that returns boolean)
    let rec convertPredicate ctx p =
        match p with
        | Condition(e) ->
            convertExpression ctx e
        | And(p1, p2) ->
            let p1 = convertPredicate ctx p1
            let p2 = convertPredicate ctx p2
            upcast Expression.AndAlso(p1, p2)
        | Or(p1, p2) ->
            let p1 = convertPredicate ctx p1
            let p2 = convertPredicate ctx p2
            upcast Expression.OrElse(p1, p2)
        | Not p -> upcast Expression.Not(convertPredicate ctx p)

    // Collect all column names in a given expression
    let rec collectExpressionColumns e = seq {
        match e with
        | GetColumn(n) -> yield n
        | Equals(e1, e2)
        | RelationalBinary(e1, e2, _)
        | NumericalBinary(e1, e2, _) ->
            yield! collectExpressionColumns e1
            yield! collectExpressionColumns e2
        | Call(_, es) ->
            for e in es do yield! collectExpressionColumns e
        | Constant _ -> () }

    // Collect all column names in a given predicate
    let rec collectPredicateColumns p = seq {
        match p with
        | Condition(e) -> yield! collectExpressionColumns e
        | And(p1, p2)
        | Or(p1, p2) ->
            yield! collectPredicateColumns p1
            yield! collectPredicateColumns p2
        | Not(p) ->
            yield! collectPredicateColumns p }

    // ------------------------------------------------------------------------------------------------
    // Putting everything together
    // ------------------------------------------------------------------------------------------------

    // Find all column names that are used in this predicate
    let collectColumns pred =
        collectPredicateColumns pred |> List.ofSeq

    let methodGetter = function
        | StringLength -> true, 1, getKnownMethod StringLength
        | StringContains -> true, 2, getKnownMethod StringContains
        | SeqContainInt -> true, 2, getKnownMethod SeqContainInt
        | SeqContainNullInt -> true, 2, getKnownMethod SeqContainNullInt
        | StringStartsWith -> true, 2, getKnownMethod StringStartsWith
        | StringEndsWith -> true, 2, getKnownMethod StringEndsWith


    // Generate `ColumnGetter` for `ConversionContext`. When accessing
    // a column, we access property of a variable of type `Team`
    let gerColumns (ty:Type)=
        [ for p in ty.GetProperties() ->
            p.Name, p.GetGetMethod() ] |> dict

    let columnGetter (ty:Type) var name  =
        let columns =  gerColumns ty
        Expression.Property(var, columns.[name]) :> Expression

    let orderByMethod nameMethod =
        typeof<Queryable>.GetMethods()
        |> Seq.filter(fun m -> m.Name = nameMethod)
        |> Seq.filter(fun m -> m.GetParameters().Length = 2)
        |> Seq.head

    let containsExpression<'a> (propertyName:string)  (propertyValue:'a) (parameter:ParameterExpression option) =
        let mi = orderByMethod "Contains"
        let parameter = defaultArg parameter (Expression.Parameter(typeof<'a>))
        let property = Expression.Property(parameter, propertyName)
        let value = Expression.Constant(propertyValue, typeof<'a>)
        let containsMethodExp = Expression.Call(property, mi, value)
        Expression.Lambda<Func<'a, bool>>(containsMethodExp, parameter)







module SortQueryModule =
    open QueryModel

    let sortLambda<'a,'b> (parameter:ParameterExpression) (name) =
        let ty = typeof<'b>
        let memberExpression = Expression.PropertyOrField(parameter, name)
        let memberExpressionConversion = Expression.Convert(memberExpression, ty)
        let body = Expression.PropertyOrField(parameter, name)
        Expression.Lambda<Func<'a, 'b>>(memberExpressionConversion, parameter)

    let sortOrderBy<'a,'b> (parameter:ParameterExpression) sortDirection (name) (q:IQueryable<'a>) =
        sortDirection
        |> function | Ascending -> q.OrderBy(sortLambda<'a,'b> parameter name)
                    | Descending -> q.OrderByDescending(sortLambda<'a,'b> parameter name)


    let sortThenBy<'a,'b> (parameter:ParameterExpression) sortDirection (name) (q:IOrderedQueryable<'a>) =
        sortDirection
        |> function | Ascending -> q.ThenBy(sortLambda<'a,'b> parameter name)
                    | Descending -> q.ThenByDescending(sortLambda<'a,'b> parameter name)

    let sortQuery (q:IQueryable<'a>) parameter (colIno, sortDirection) =
        match colIno.ColumnType with
        | x when x = typeof<string> -> sortOrderBy<'a, string> parameter sortDirection colIno.ColumnName q
        | x when x = typeof<int> ->  sortOrderBy<'a, int> parameter sortDirection colIno.ColumnName q
        | x when x = typeof<float> ->   sortOrderBy<'a, float> parameter sortDirection colIno.ColumnName q
        | x when x = typeof<DateTime> ->   sortOrderBy<'a, DateTime> parameter sortDirection colIno.ColumnName q
        | _ -> failwith(sprintf "Column Type %s for sorting not supported" colIno.ColumnType.Name)

    let sortOrderedQuery (q:IOrderedQueryable<'a>) parameter (colIno, sortDirection) =
        match colIno.ColumnType with
        | x when x = typeof<string> -> sortThenBy<'a, string> parameter sortDirection colIno.ColumnName q
        | x when x = typeof<int> ->  sortThenBy<'a, int> parameter sortDirection colIno.ColumnName q
        | x when x = typeof<float> ->   sortThenBy<'a, float> parameter sortDirection colIno.ColumnName q
        | x when x = typeof<DateTime> ->   sortThenBy<'a, DateTime> parameter sortDirection colIno.ColumnName q
        | _ -> failwith(sprintf "Column Type %s for sorting not supported" colIno.ColumnType.Name)

    let getSorting (sortBy:SortBy) =
                match sortBy with | SortBy(s) -> s

//
//    let [<Literal>] OrderByLiteral = "OrderBy"
//    let [<Literal>] OrderByDescendingLiteral = "OrderByDescending"
//    let [<Literal>] ThenByLiteral = "ThenBy"
//    let [<Literal>] ThenByDescendingLiteral = "ThenByDescending"
//
//    let orderOptions =
//         [  (OrderBy, lazy(orderByMethod OrderByLiteral))
//            (OrderByDescending, lazy(orderByMethod OrderByDescendingLiteral))
//            (ThenBy, lazy(orderByMethod ThenByLiteral))
//            (ThenByDescending, lazy(orderByMethod ThenByDescendingLiteral))
//         ] |> dict

//    let sortEval (q:IQueryable<'a>) parameter (columnInfo:ColumnInfo) orderOption =
//        let {ColumnName=columnName; ColumnType=columnType} = columnInfo
//        let orderBy = Expression.Property(parameter,columnName)
//        let lambda = Expression.Lambda(orderBy, [| parameter |])
//        let genMethod = orderOptions.[orderOption].Value.MakeGenericMethod([|typeof<'a>;columnType|])
//        genMethod.Invoke(q, [|q; lambda |]) :?> IQueryable<'a>


//    let sortEval (q:IQueryable<'a>) parameter (columnInfo:ColumnInfo) orderOption =
//        let {ColumnName=columnName; ColumnType=columnType} = columnInfo
//        let orderBy = Expression.Property(parameter,columnName)
//        let lambda = Expression.Lambda(orderBy, [| parameter |])
//
//        let genMethod = orderOptions.[orderOption].Value.MakeGenericMethod([|typeof<'a>;columnType|])
//        genMethod.Invoke(q, [|q; lambda |]) :?> IQueryable<'a>
//
module PagingQueryModule =
    open QueryModel

    let page p (query:IQueryable<'a>)=
        match p with
        | Take(n) -> query.Take(n)
        | Skip(n) -> query.Skip(n)



module ExpUtil =
    let rec translateSimpleExpr expr =
      match expr with
      | Patterns.Var(var) ->
          // Variable access
          Expression.Variable(var.Type, var.Name) :> Expression
      | Patterns.PropertyGet(Some inst, pi, []) ->
          // Getter of an instance property
          let instExpr = translateSimpleExpr inst
          Expression.Property(instExpr, pi) :> Expression
      | Patterns.Call(Some inst, mi, args) ->
          // Method call - translate instance & arguments recursively
          let argsExpr = Seq.map translateSimpleExpr args
          let instExpr = translateSimpleExpr inst
          Expression.Call(instExpr, mi, argsExpr) :> Expression
      | Patterns.Call(None, mi, args) ->
          // Static method call - no instance
          let argsExpr = Seq.map translateSimpleExpr args
          Expression.Call(mi, argsExpr) :> Expression
      | _ -> failwith "not supported"

    /// Translates a simple F# quotation to a lambda expression
    let translateLambda (expr:Expr<'T -> 'R>) =
      match expr with
      | Patterns.Lambda(v, body) ->
          // Build LINQ style lambda expression
          let bodyExpr = translateSimpleExpr body
          let paramExpr = Expression.Parameter(v.Type, v.Name)
          Expression.Lambda<Func<'T, 'R>>(paramExpr)
      | _ -> failwith "not supported"

    let IsOption (stype: System.Type) = stype.Name = "FSharpOption`1"
    let inline application prms expr = Expr.Application(expr, prms)
    let inline coerse typ expr = Expr.Coerce(expr, typ)
    let inline newrec typ args = Expr.NewRecord(typ, args)
    let inline inList (membr: Expr<'a -> 'b>) (values: 'b list) : Expr<'a -> bool> =
        match membr with
        | Lambda (_, PropertyGet _) ->
            match values with
            | [] -> <@ fun _ -> true @>
            | _ -> values |> Seq.map (fun v -> <@ fun a -> (%membr) a = v @>) |> Seq.reduce (fun a b -> <@ fun x -> (%a) x || (%b) x @>)
        | _ -> failwith "Expression has to be a member"

    let (|IsMapType|_|) (t: Type) =
        if t.IsGenericType && t.GetGenericTypeDefinition() = typedefof<Map<_,_>> then Some t
        else None

    let rec copyThing (mtype: Type) : Expr =
        match mtype with
        | _ when FSharpType.IsRecord mtype -> genRecordCopier mtype
        | _ when FSharpType.IsUnion mtype  -> genUnionCopier mtype
        | _ when mtype.IsValueType || mtype = typeof<String> -> getIdFunc mtype
        | _ when mtype.IsArray -> genArrayCopier mtype
        | IsMapType t -> getIdFunc mtype
        | _ when mtype = typeof<System.Object> -> getIdFunc mtype
        | _ -> failwithf "Unexpected Type: %s" (mtype.ToString())

    and X<'T> : 'T = Unchecked.defaultof<'T>

    and getMethod =
        function
        | Patterns.Call (_, m, _) when m.IsGenericMethod -> m.GetGenericMethodDefinition()
        | Patterns.Call (_, m, _) -> m
        | _ -> failwith "Incorrect getMethod Pattern"

    and getIdFunc itype =
        let arg = Var("x", itype, false)
        let argExpr = Expr.Var(arg)
        let func =
            let m = (getMethod <@ id X @>).MakeGenericMethod([|itype|])
            Expr.Call(m, [argExpr])
        Expr.Lambda(arg, func)

    and genRecordCopier (rtype: Type) : Expr =
        let arg = Var("x", rtype, false)
        let argExpr = Expr.Var(arg)
        let newrec =
            FSharpType.GetRecordFields(rtype) |> Array.toList
            |> List.map (fun field -> genFieldCopy argExpr field)
            |> newrec rtype
        Expr.Lambda(arg, newrec)

    and genFieldCopy argExpr (field: PropertyInfo) : Expr =
        let pval = Expr.PropertyGet(argExpr, field)
        copyThing field.PropertyType |> application pval

    and genArrayCopier (atype : Type) : Expr =
        let etype = atype.GetElementType()
        let copyfun = copyThing etype

        let arg = Var("arr", atype, false)
        let argExpr = Expr.Var(arg)

        let func =
            let m = (getMethod <@ Array.map X X @>).MakeGenericMethod([|etype; etype|])
            Expr.Call(m, [copyfun; argExpr])

        Expr.Lambda(arg, func)

    and genUnionCopier (utype: Type) : Expr =
        let cases = FSharpType.GetUnionCases utype
        // if - union case - then - copy each field into new case - else - next case

        let arg = Var("x", utype, false)
        let useArg = Expr.Var(arg)

        let genCaseTest case = Expr.UnionCaseTest (useArg, case)

        let makeCopyCtor (ci: UnionCaseInfo) =
            let copiedMembers = [ for field in ci.GetFields() -> genFieldCopy useArg field ]
            Expr.NewUnionCase(ci, copiedMembers)

        let genIf ifCase thenCase elseCase = Expr.IfThenElse(ifCase, thenCase, elseCase)

        let typedFail (str: string) =
            let m = (getMethod <@ failwith str @>).MakeGenericMethod([|utype|])
            Expr.Call(m, [ <@ str @> ])

        let nestedIfs =
            cases
            |> Array.map (fun case -> genIf (genCaseTest case) (makeCopyCtor case))
            |> Array.foldBack (fun iff st -> iff st) <| (typedFail "Unexpected Case in Union")

        Expr.Lambda(arg, nestedIfs)

    let toLinq<'I,'O> (expr: Expr<'I -> 'O>) =
        let linq = Microsoft.FSharp.Linq.RuntimeHelpers.LeafExpressionConverter.QuotationToExpression expr
        let call = linq :?> MethodCallExpression
        let lambda  = call.Arguments.[0] :?> LambdaExpression
        Expression.Lambda<Func<'I,'O>>(lambda.Body, lambda.Parameters)

    let genrateRecordDeepCopyFunction<'T> () : ('T -> 'T) =
        let expr = genRecordCopier typeof<'T>
        let castExpr : Expr<'T -> 'T> = expr |> Expr.Cast
        let compiledExpr = (castExpr |> toLinq).Compile()
        fun (v : 'T) -> compiledExpr.Invoke(v)

module QuotationPrinter =
    let rec print depth (expr:Expr) =
        match expr with
        | Patterns.Value (v, typ) -> sprintf "%A" v
        | Patterns.Var v ->  sprintf "%s:%s" v.Type.Name v.Name
        | Patterns.NewUnionCase (uci, args) ->
            sprintf "%s(%s)" uci.Name (printArgs depth args)
        | Patterns.NewArray (_,args) ->
            sprintf "[%s]" (printArgs depth args)
        | Patterns.NewRecord (_,args) ->
            sprintf "{%s}" (printArgs depth args)
        | Patterns.NewTuple args ->
            sprintf "(%s)" (printArgs depth args)
        | Patterns.NewObject (ci, args) ->
            sprintf "new %s(%s)" ci.Name (printArgs depth args)
        | Patterns.Call (Some (Patterns.ValueWithName(_,_,instance)), mi, args) ->
            sprintf "%s.%s(%s)" instance mi.Name (printArgs (depth + 1) args)
        | Patterns.Call (None, mi, args) ->
            sprintf "%s(%s)" mi.Name (printArgs (depth + 1) args)
        | Patterns.Lambda (var, body) ->
            sprintf "(λ %s -> %s)" (print 0 (Expr.Var var)) (printArgs (depth + 1) [body])
        | Patterns.Let (var, bind, body) ->
            sprintf "let %s = %s in\r\n%*s%s" (print 0 (Expr.Var var)) (print 0 bind) ((depth - 1) * 4) "" (print depth body)
        | Patterns.PropertyGet (Some(var), pi, args) ->
            sprintf "%s.%s" (print 0 var) pi.Name
        | Patterns.PropertySet (Some(var), pi, args, value) ->
            sprintf "%s.%s <- %s" (print 0 var) pi.Name (print depth value)
        | Patterns.Sequential (x,y) ->
            sprintf "%s; %s" (print depth x) (print depth y)
        | a -> failwithf "Unknown patterns %A" a

    and printArgs depth args =
        match args with
        | [a] -> sprintf "\r\n%*s%s\r\n%*s" (depth * 4) "" (print (depth + 1) a) (depth * 4) ""
        | a ->
            sprintf "\r\n%*s%s" (depth * 4) "" (String.Join(sprintf ",\r\n%*s" (depth * 4) "",List.map (print (depth + 1)) a))