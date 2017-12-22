#r "../Lib/EntityFramework.dll"
#r "../Lib/EntityFramework.SqlServer.dll"
#r "System.Data.dll"
#r "System.Data.Linq.dll"
#r "System.ComponentModel.DataAnnotations.dll"
#r "System.Data.Entity.dll"
#load "medals.fs"

open System
open System.Linq
open System.Reflection
open System.Linq.Expressions
open FSharp.Quotations
open FSharp.Quotations.Patterns
open Microsoft.FSharp.Reflection
open Microsoft.FSharp.Quotations
open System.Data.Entity

let select = typeof<Enumerable>.GetMethods() |> Seq.find (fun m -> m.Name = "SingleOrDefault")


// ------------------------------------------------------------------------------------------------
// Domain model
//
// In the model everything is just `QueryExpression`, which represents
// some calculation (using one or more columns) that produces a value of type `int` (but it could be `float` too),
// `bool` or `string`. The expressions that return `bool` are used in filtering
// The `QueryTransform` is a single transformation applied on the whole input table.
// This can sort the table, filter the table or group table data using a key:
//
//  - `Filter` takes a `QueryExpression` (which should represent a computation returning bool)
//
//  - `SortBy` takes one or more expressions (that represent sort keys) together with the
//    sorting direction. The first one is mapped to `OrderBy` and all others are mapped to `ThenBy`
//
//  - `GroupBy` takes a key selector which gives us the grouping key (e.g. country name of an
//    athlete) and a number of "aggregations". Those describe what to do with the columns in the
//    group. `Sum(GetColumn("Gold"))` means "sum the values of 'Gold' column for the group"
//    (I only included sum, but you could add averaging etc. You'll probably also need
//    something like `Enumerable.SingleOrDefault` (to return value assuming it is same for
//    every item in the group).


//
// ------------------------------------------------------------------------------------------------
// The code takes a `Predicate` and generates
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
module QueryModule =
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

    type UnaryOperator =
        | Not

    // Numerical operators. Take two numerical
    // expressions and produce new numerical expression.
    type NumericalOperator =
        | Plus
        | Minus
        | Divide
        | Mul

    // Assuming the model supports a small fixed number of other methods, we list them
    // in this DU - for now we can check string length & string contains
    // (By representing the methods as DU cases and not, e.g. quotation, we can easily
    // pattern match on them)
    // it might make sense to just store their `MethodInfo`
    type KnownMethod = // TODO more here as it goes
        | StringContains
        | StringLength
        | StringStartsWith
        | StringEndsWith
        | SeqContainInt
        | SeqContainNullInt

    // Relational operators (just > and <). Note that `=` is included in `QueryExpression`
    // as a separate case. This is because `<` and `>` can be applied only on numbers, but
    // `=` can be used to test equality of anything (numbers, but also strings)
    type RelationOperator =
        | LessThan
        | GreaterThan
        | LessThanOrEqual
        | GreaterThanOrEqual
    type BinaryOperator =
        | Plus
        | Minus
        | Divide
        | Mul
        | Equals
        | RelationOperator of RelationOperator
//        | LessThan
//        | GreaterThan
//        | LessThanOrEqual
//        | GreaterThanOrEqual
        | And
        | Or

    // Represents expression, which can be evaluated to value of various types
    type QueryExpression =
        | GetColumn of ColumnName
        | Binary of QueryExpression * QueryExpression * BinaryOperator
        | Unary of QueryExpression * UnaryOperator
        | Call of KnownMethod * QueryExpression list
        | Constant of QueryValue
        | ContainsElements of CollectionReference * QueryExpression

    and ColumnName = string

    and CollectionReference =
        | InMemoryInts of SeqInts
        | InMemoryNullInts of SeqNullInts
    and SeqNullInts = System.Collections.Generic.IEnumerable<System.Nullable<int>>
    and SeqInts = System.Collections.Generic.IEnumerable<int>



    // [MULTIKEY] We now have more than one grouping keys and so when we
    // want to get one of the multiple keys, we need to specify which
    // one we want to get - here, we just use an int, which can be easily
    // mapped to accessing `ElementN` of the `GroupTuple` type.
    type Aggregation =
        | Sum of QueryExpression
        | SingleOrDefault of QueryExpression
        | Key of int //GroupKey
        //| CountAll
        //| CountDistinct of string
        //| ReturnUnique of string
        //| ConcatValues of string

    // Sorting
    type ColumnInfo =
        { ColumnName : ColumnName
          ColumnType : Type }

    and SortDirection =
        | Ascending
        | Descending

    // and SortBy = SortBy of (ColumnInfo * SortDirection) list
    type Paging =
        | Take of int
        | Skip of int

    // When converting, the caller needs to tell us how to convert property access
    // and how to implement methods. `MethodGetter` returns whether the method is
    // instance method, number of parameters (including instance parameter) and
    // the `MethodInfo` to be used in the expression tree.
    type FilterQuery = QueryExpression

    type GroupQuery = QueryExpression list * (ColumnName * Aggregation) list

    type SortQuery = (QueryExpression * SortDirection) list

    type QueryTransform =
        | Filter of FilterQuery
        | SortBy of SortQuery
        | GroupBy of GroupQuery
        | Paging of Paging list

    and QueryPredicate =
        | Predicate of QueryExpression
        | Nil
        | Wrong of string option

    type ConversionContext =
        { ColumnGetter : ColumnName -> Expression
          MethodGetter : KnownMethod -> (bool * int * MethodInfo) }

module QueryFunctionModule =
    open QueryModule

    let inline ZeroVal<'a when 'a : struct and 'a : (new : unit -> 'a) and 'a :> ValueType>() = System.Nullable<'a>()
    let inline Val<'a when 'a : struct and 'a : (new : unit -> 'a) and 'a :> ValueType>(v : 'a) = System.Nullable<'a>(v)

    // Generate `MethodGetter` for mapping method names to MethodInfo
    let geSeqMethod ty name =
        typeof<Enumerable>.GetMethods()
        |> Seq.filter (fun m -> m.Name = name)
        |> Seq.find (fun m -> m.GetParameters().Length = 2)
        |> fun m -> m.MakeGenericMethod([| ty |])

    let seqContainMethod ty = geSeqMethod ty "Contains"

    let getKnownMethod this =
        match this with
        | StringLength -> typeof<string>.GetProperty("Length").GetGetMethod()
        | StringContains -> typeof<string>.GetMethod("Contains")
        | StringStartsWith -> typeof<string>.GetMethod("StartsWith")
        | StringEndsWith -> typeof<string>.GetMethod("EndsWith")
        | SeqContainInt -> seqContainMethod typeof<int>
        | SeqContainNullInt -> seqContainMethod typeof<int Nullable>

    //    It is possible to support <@@> - Transform `QueryExpression` into LINQ `Expression`.
    //    let rec convertExpression ctx e =
    //        match e with
    //        | Constant(Number n) -> <@@ n @@>
    //        | Constant(Boolean n) -> <@@ n @@>
    //        | NumericalBinary(e1, e2, NumericalOperator.Plus) -> <@@ %%(convertExpression ctx e1) + %%(convertExpression ctx e2) @@>
    //        | Call(KnownMethod.StringContains, [e1; e2]) -> <@@ (%%(convertExpression ctx e1):string).Contains(%%(convertExpression ctx e2)) @@>
    let equalWithNullValue (e1 : Expression) (e2 : Expression) =
        let hasValueExp = Expression.Property(e2, "HasValue")
        let valueExp = Expression.Property(e2, "Value")
        let equal = Expression.Equal(valueExp, e1)
        Expression.AndAlso(hasValueExp, equal)

    let equalbetweenNullValues (e1 : Expression) (e2 : Expression) =
        let hasValueExp1 = Expression.Property(e1, "HasValue")
        let valueExp1 = Expression.Property(e1, "Value")
        let hasValueExp2 = Expression.Property(e2, "HasValue")
        let valueExp2 = Expression.Property(e2, "Value")
        let equal = Expression.Equal(valueExp1, valueExp2)
        Expression.AndAlso(Expression.AndAlso(hasValueExp2, hasValueExp1), equal)

    let orderByMethod nameMethod =
        typeof<Queryable>.GetMethods()
        |> Seq.filter (fun m -> m.Name = nameMethod)
        |> Seq.filter (fun m -> m.GetParameters().Length = 2)
        |> Seq.head

    let containsExpression<'a> (propertyName : string) (propertyValue : 'a) (parameter : ParameterExpression option) =
        let mi = orderByMethod "Contains"
        let parameter = defaultArg parameter (Expression.Parameter(typeof<'a>))
        let property = Expression.Property(parameter, propertyName)
        let value = Expression.Constant(propertyValue, typeof<'a>)
        let containsMethodExp = Expression.Call(property, mi, value)
        Expression.Lambda<Func<'a, bool>>(containsMethodExp, parameter)

    let property (propertyName : string) (param : ParameterExpression) =
        // Builds up a MemberExpression that navigates to
        // nested properties
        propertyName.Split([| '.' |])
        |> Seq.fold (fun state property -> Expression.Property(state, property) :> Expression) (param :> Expression)

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
                failwith
                    (sprintf "Error when converiting call to '%s'. Expected '%d' arguments but got '%d'" mi.Name
                         argCount (List.length args))
            let args = List.map (convertExpression ctx) args
            if not isInstance then upcast Expression.Call(mi, args)
            else upcast Expression.Call(List.head args, mi, List.tail args)
        | Binary(e1, e2, op) ->
            // Generate equality test - this should work for any
            // type, but LINQ's `Expression.Equal` seems to be clever
            // enough to handle that for us :-)
            let e1 = convertExpression ctx e1
            let e2 = convertExpression ctx e2
            match op with
            | And -> upcast Expression.AndAlso(e1, e2)
            | Or -> upcast Expression.OrElse(e1, e2)
            | RelationOperator rop ->
                match rop with
                | LessThan -> upcast Expression.LessThan(e1, e2)
                | LessThanOrEqual -> upcast Expression.LessThanOrEqual(e1, e2)
                | GreaterThan -> upcast Expression.GreaterThan(e1, e2)
                | GreaterThanOrEqual -> upcast Expression.GreaterThanOrEqual(e1, e2)
            //      | Equals((Constant(NullValue(_)) as e1), (Constant(NullValue(_)) as e2)) ->
            //            let e1 = convertExpression ctx e1
            //            let e2 = convertExpression ctx e2
            //             upcast equalbetweenNullValues e2 e1
            //      | Equals(e1, (Constant(NullValue(v)) as e2)) ->
            //            let e1 = convertExpression ctx e1
            //            let e2 = convertExpression ctx e2
            //            upcast equalWithNullValue e1 e2
            //      | Equals((Constant(NullValue(v)) as e1), e2) ->
            //            let e1 = convertExpression ctx e1
            //            let e2 = convertExpression ctx e2
            //            upcast equalWithNullValue e2 e1
            | Equals -> upcast Expression.Equal(e1, e2)
            | Plus -> upcast Expression.Add(e1, e2)
            | Minus -> upcast Expression.Subtract(e1, e2)
            | Divide -> upcast Expression.Divide(e1, e2)
            | Mul -> upcast Expression.Multiply(e1, e2)
        | Unary(e, op) ->
            let e = convertExpression ctx e
            match op with
            | Not -> upcast Expression.Not(e)

    // ------------------------------------------------------------------------------------------------
    // Creating functions - Also same as before. This is a helper that creates a function from
    // an expression using `convertExpression` (or other given conversion function)
    // ------------------------------------------------------------------------------------------------
    let methodGetter =
        function
        | StringLength -> true, 1, getKnownMethod StringLength
        | StringContains -> true, 2, getKnownMethod StringContains
        | SeqContainInt -> true, 2, getKnownMethod SeqContainInt
        | SeqContainNullInt -> true, 2, getKnownMethod SeqContainNullInt
        | StringStartsWith -> true, 2, getKnownMethod StringStartsWith
        | StringEndsWith -> true, 2, getKnownMethod StringEndsWith

    let columnGetter<'T> var name =
        let columns = // TODO MEMOIZE
            [ for p in typeof<'T>.GetProperties() -> p.Name, p.GetGetMethod() ]
            |> dict
        Expression.Property(var, columns.[name]) :> Expression

    // Generate `ColumnGetter` for `ConversionContext`. When accessing
    // a column, we access property of a variable of type `Team`
    let gerColumns (ty : Type) =
        [ for p in ty.GetProperties() -> p.Name, p.GetGetMethod() ]
        |> dict

    let makeFunction<'T, 'R> f e : Expression<Func<'T, 'R>> =
        let var = Expression.Parameter(typeof<'T>)

        let ctx =
            { MethodGetter = methodGetter
              ColumnGetter = columnGetter<'T> var }

        let body = f ctx e
        Expression.Lambda<Func<'T, 'R>>(body, var)



    // [MULTIKEY] When creating function for GroupBy, we also return `types` and so we
    // need a little different `makeFunction` operation (that just propagates the additional
    // resulting types). There is some code duplication here, but not too bad if we only
    // need these two functions to make it work!
    let makeGroupFunction<'T, 'R> f e : _ * Expression<Func<'T, 'R>> =
      let var = Expression.Parameter(typeof<'T>)
      let ctx =
        { MethodGetter = methodGetter
          ColumnGetter = columnGetter<'T> var }
      let types, body = f ctx e
      types, Expression.Lambda<Func<'T, 'R>>(body, var)



    /// This returns default (empty) value of any type that may appear in the DB "entity"
    let getDefaultValue typ =
        if typ = typeof<int> then Expression.Constant(0)
        elif typ = typeof<float> then Expression.Constant(0.0)
        elif typ = typeof<DateTime> then Expression.Constant(DateTime.MinValue) //
        elif typ = typeof<string> then Expression.Constant(null, typeof<string>)
        else failwithf "Cannot get default value of type %s" typ.FullName


    // Collect all column names in a given expression
    let rec collectExpressionColumns e =
        seq {
            match e with
            | GetColumn(n) -> yield n
            | Binary(e1, e2, _) ->
                yield! collectExpressionColumns e1
                yield! collectExpressionColumns e2
            | ContainsElements(_, es) -> yield! collectExpressionColumns es
            | Call(_, es) ->
                for e in es do
                    yield! collectExpressionColumns e
            | Unary(e, _) -> yield! collectExpressionColumns e
            | Constant _ -> ()
        }

    // Find all column names that are used in a given query
    let collectQueryColumns q =
        let rec collectQueryColumns (q : QueryTransform) =
            seq {
                match q with
                | Filter(e) -> yield! collectExpressionColumns e
                | SortBy(qs) ->
                    for e in qs |> Seq.map fst do
                        yield! collectExpressionColumns e

                | GroupBy(e, _) ->
                    for ex in e do
                         yield! collectExpressionColumns ex
                | Paging(_) -> ()
            }
        collectQueryColumns q |> List.ofSeq

// ------------------------------------------------------------------------------------------------
// Dealing with sorting and grouping
//
// When calling `OrderBy` (and also `GroupBy`), we need to know the type of the key we are
// using statically. This is difficult, because the key may be `string` or `int`, or perhaps
// something else (but we only have a few types). Here, we define two types that look like
// anonymous C# types (so that LINQ understands them) and use them as sort/group keys. They
// contain property for each type we may need. We only initialize one of the properties, so
// this works fine and we have static type for the key. In C#, this is like writing:
//
//    input.OrderBy(m => new SortKey { String = m.Player_Name })
//
// ------------------------------------------------------------------------------------------------
module SortingModule =
    open QueryFunctionModule
    open QueryModule

    type SortKey() =
        [<DefaultValue>]
        val mutable _string : string
        [<DefaultValue>]
        val mutable _number : int
        [<DefaultValue>]
        val mutable _numberF : float
        [<DefaultValue>]
        val mutable _date : DateTime

        member x.String
            with get () = x._string
            and set (v) = x._string <- v

        member x.Number
            with get () = x._number
            and set (v) = x._number <- v

        member x.NumberF
            with get () = x._numberF
            and set (v) = x._numberF <- v

        member x.Date
            with get () = x._date
            and set (v) = x._date <- v

    let orderByMethod nameMethod =
        typeof<Queryable>.GetMethods()
        |> Seq.filter (fun m -> m.Name = nameMethod)
        |> Seq.filter (fun m -> m.GetParameters().Length = 2)
        |> Seq.head

    let sortLambda<'a, 'b> (parameter : ParameterExpression) (name) =
        let ty = typeof<'b>
        let memberExpression = Expression.PropertyOrField(parameter, name)
        let memberExpressionConversion = Expression.Convert(memberExpression, ty)
        let body = Expression.PropertyOrField(parameter, name)
        Expression.Lambda<Func<'a, 'b>>(memberExpressionConversion, parameter)

    let sortOrderBy<'a, 'b> (parameter : ParameterExpression) sortDirection (name) (q : IQueryable<'a>) =
        sortDirection |> function
        | Ascending -> q.OrderBy(sortLambda<'a, 'b> parameter name)
        | Descending -> q.OrderByDescending(sortLambda<'a, 'b> parameter name)

    let sortThenBy<'a, 'b> (parameter : ParameterExpression) sortDirection (name) (q : IOrderedQueryable<'a>) =
        sortDirection |> function
        | Ascending -> q.ThenBy(sortLambda<'a, 'b> parameter name)
        | Descending -> q.ThenByDescending(sortLambda<'a, 'b> parameter name)

    let sortQuery (q : IQueryable<'a>) parameter (colIno, sortDirection) =
        match colIno.ColumnType with
        | x when x = typeof<string> -> sortOrderBy<'a, string> parameter sortDirection colIno.ColumnName q
        | x when x = typeof<int> -> sortOrderBy<'a, int> parameter sortDirection colIno.ColumnName q
        | x when x = typeof<float> -> sortOrderBy<'a, float> parameter sortDirection colIno.ColumnName q
        | x when x = typeof<DateTime> -> sortOrderBy<'a, DateTime> parameter sortDirection colIno.ColumnName q
        | _ -> failwith (sprintf "Column Type %s for sorting not supported" colIno.ColumnType.Name)

    let sortOrderedQuery (q : IOrderedQueryable<'a>) parameter (colIno, sortDirection) =
        match colIno.ColumnType with
        | x when x = typeof<string> -> sortThenBy<'a, string> parameter sortDirection colIno.ColumnName q
        | x when x = typeof<int> -> sortThenBy<'a, int> parameter sortDirection colIno.ColumnName q
        | x when x = typeof<float> -> sortThenBy<'a, float> parameter sortDirection colIno.ColumnName q
        | x when x = typeof<DateTime> -> sortThenBy<'a, DateTime> parameter sortDirection colIno.ColumnName q
        | _ -> failwith (sprintf "Column Type %s for sorting not supported" colIno.ColumnType.Name)

    let getSorting =
        function
        | SortBy(s) -> s
        | _ -> failwith ("Only Sorting allowed")

    /// This is a wrapper over `convertExpression` that first does whatever `convertExpression`
    /// does (this would generate `m.Player_Name` in the above example) and then wraps it
    /// to produce a LINQ expression returning `SortKey`.
    let convertSortExpression ctx e =
        let e = convertExpression ctx e
        // MemberInit represents C# type initializer - it takes constructor
        // and one or more member assignments. Here, we always give it just
        // one of the two assignments (for string or for number)
        Expression.MemberInit(Expression.New(typeof<SortKey>.GetConstructors().[0]),
                              [ if e.Type = typeof<string> then
                                    yield Expression.Bind(typeof<SortKey>.GetProperty("String"), e) :> MemberBinding
                                elif e.Type = typeof<int> then
                                    yield Expression.Bind(typeof<SortKey>.GetProperty("Number"), e) :> MemberBinding
                                elif e.Type = typeof<float> then
                                    yield Expression.Bind(typeof<SortKey>.GetProperty("NumberF"), e) :> MemberBinding
                                elif e.Type = typeof<DateTime> then
                                    yield Expression.Bind(typeof<SortKey>.GetProperty("Date"), e) :> MemberBinding
                                else failwith "Not supported type of sort key" ]) :> Expression

    // Call appropriate ThenBy operation on an IQueryable...
    let thenSortBy (input : IOrderedQueryable<'T>) =
        function
        | e, Ascending -> input.ThenBy(makeFunction<'T, SortKey> convertSortExpression e)
        | e, Descending -> input.ThenByDescending(makeFunction<'T, SortKey> convertSortExpression e)

module GroupingModule =
    open QueryFunctionModule
    open QueryModule

    type GroupKey() =
        [<DefaultValue>]
        val mutable _string : string
        [<DefaultValue>]
        val mutable _number : int
        [<DefaultValue>]
        val mutable _numberF : float
        [<DefaultValue>]
        val mutable _date : DateTime


        member x.String
            with get () = x._string
            and set (v) = x._string <- v

        member x.Number
            with get () = x._number
            and set (v) = x._number <- v

        member x.NumberF
            with get () = x._numberF
            and set (v) = x._numberF <- v

        member x.Date
            with get () = x._date
            and set (v) = x._date <- v


    type GroupTuple() =
      [<Microsoft.FSharp.Core.DefaultValue>] val mutable _element0 : GroupKey
      [<Microsoft.FSharp.Core.DefaultValue>] val mutable _element1 : GroupKey
      [<Microsoft.FSharp.Core.DefaultValue>] val mutable _element2 : GroupKey
      [<Microsoft.FSharp.Core.DefaultValue>] val mutable _element3 : GroupKey
      [<Microsoft.FSharp.Core.DefaultValue>] val mutable _element4 : GroupKey
      [<Microsoft.FSharp.Core.DefaultValue>] val mutable _element5 : GroupKey
      [<Microsoft.FSharp.Core.DefaultValue>] val mutable _element6 : GroupKey
      [<Microsoft.FSharp.Core.DefaultValue>] val mutable _element7 : GroupKey
      [<Microsoft.FSharp.Core.DefaultValue>] val mutable _element8 : GroupKey
      member x.Element0 with get() = x._element0 and set(v) = x._element0 <- v
      member x.Element1 with get() = x._element1 and set(v) = x._element1 <- v
      member x.Element2 with get() = x._element2 and set(v) = x._element2 <- v
      member x.Element3 with get() = x._element3 and set(v) = x._element3 <- v
      member x.Element4 with get() = x._element4 and set(v) = x._element4 <- v
      member x.Element5 with get() = x._element5 and set(v) = x._element5 <- v
      member x.Element6 with get() = x._element6 and set(v) = x._element6 <- v
      member x.Element7 with get() = x._element7 and set(v) = x._element7 <- v
      member x.Element8 with get() = x._element8 and set(v) = x._element8 <- v





    // When converting grouping, things are more complicated. We will need to generate
    // LINQ code at runtime using things like `Sum` and `Select`...
    let sumInt = typeof<Enumerable>.GetMethod("Sum", [| typeof<seq<int>> |]) // TODO Memoize ?
    let select = typeof<Enumerable>.GetMethods() |> Seq.find (fun m -> m.Name = "Select")

    let callMethod<'T> e =
        let var = Expression.Parameter(typeof<'T>)
        let ctx =
            { MethodGetter = methodGetter
              ColumnGetter = columnGetter<'T> var }
        convertExpression ctx e, var




    // We take `input` (variable representing the group), `keyType`, which is the type
    // of the value used as a grouping key, `keyExpr` which is an expression for accessing
    // the grouping key using `g.Key` and the desired aggregation.
    //
    // [MULTIKEY] Now we also need to take types of individual keys rather than just one `keyType`
    let convertAggregation<'T> input (keyTypes:_ list) (keyExpr:Expression) agg =
        match agg with


       // Recall that the type of grouping key is `GroupKey`, so we need to extract the original value
       // [MULTIKEY] If we want to acces Nth key of type `string`, we need to generate accessor
       // expression of the form `g.Key.ElementN.String` (because `g.Key` is of type `GroupTuple` now!)
       | Key n when keyTypes.[n] = typeof<string> ->
          Expression.Property
            ( Expression.Property(keyExpr, "Element" + string n),
              typeof<GroupKey>.GetProperty("String").GetMethod) :> Expression
       | Key n when keyTypes.[n] = typeof<int> ->
          Expression.Property
            ( Expression.Property(keyExpr, "Element" + string n),
              typeof<GroupKey>.GetProperty("Number").GetMethod) :> Expression


       | Key n when keyTypes.[n] = typeof<float> ->
          Expression.Property
            ( Expression.Property(keyExpr, "Element" + string n),
              typeof<GroupKey>.GetProperty("NumberF").GetMethod) :> Expression
       | Key n when keyTypes.[n] = typeof<DateTime> ->
          Expression.Property
            ( Expression.Property(keyExpr, "Element" + string n),
              typeof<GroupKey>.GetProperty("Date").GetMethod) :> Expression

        | Key _ -> failwith "Unsupported key type"
        // We want to sum values produced by the given expression for all rows in the group
        // the following generates `g.Select(var => <...compile e using var...>).Sum()`
        | Sum(e) ->
            let var = Expression.Parameter(typeof<'T>)

            let ctx =
                { MethodGetter = methodGetter
                  ColumnGetter = columnGetter<'T> var }

            let e = convertExpression ctx e
            let e = Expression.Call(select.MakeGenericMethod(typeof<'T>, e.Type), input, Expression.Lambda(e, var))
            if e.Type = typeof<seq<int>> then upcast Expression.Call(sumInt, e)
            else failwith "Can only sum ints"




    // [MULTIKEY] This now generates an expression that returns one of the multiple
    // grouping keys as `GroupKey`. Say we want to use `Athlete` as one of the key -
    // the following builds LINQ expression representing `new GroupKey { String = g.Athlete }`
    // We also need to keep track of the types of the individual keys, so this function
    // now returns the original type of the key, together with the expression.
    let convertSingleGroupExpression ctx e =
      let e = convertExpression ctx e
      e.Type,
      Expression.MemberInit
        ( Expression.New(typeof<GroupKey>.GetConstructors().[0]),
          [ let s = if e.Type = typeof<string> then e else upcast getDefaultValue typeof<string>
            yield Expression.Bind(typeof<GroupKey>.GetProperty("String"), s) :> MemberBinding
            let n = if e.Type = typeof<int> then e else upcast getDefaultValue typeof<int>
            yield Expression.Bind(typeof<GroupKey>.GetProperty("Number"), n) :> MemberBinding ] ) :> Expression

    // [MULTIKEY] I had to change the above function so that it always generates assignment
    // for all of the `GroupKey` properties, using default values for the other ones. This is
    // the same as the other use of `getDefaultValue` below. Otherwise, LINQ does not know how
    // to translate our expression and tells us "System.NotSupportedException: The type 'GroupKey'
    // appears in two structurally incompatible initializations within a single LINQ to Entities query."
    // (but the above solution works fine!)


    // [MULTIKEY] This now generates an expression that returns `GroupTuple` containing all the keys.
    // If we are grouping by `Athlete` and `Year`, this will produce a LINQ expression like this:
    //
    //     new GroupTuple { Element0 = new GroupKey { String = g.Athlete }
    //                      Element1 = new GroupKey { Number = g.Year } }
    //
    let convertGroupExpression ctx es =
      let types, es = List.map (convertSingleGroupExpression ctx) es |> List.unzip
      types,
      Expression.MemberInit
        ( Expression.New(typeof<GroupTuple>.GetConstructors().[0]),
          [ for i, e in Seq.indexed es do
              yield Expression.Bind(typeof<GroupTuple>.GetProperty("Element" + string i), e) :> MemberBinding ] ) :> Expression




//    //let convertGroupExpression : ctx QueryExpres list ->
//
//    let convertGroupExpression ctx e =
//        let e = convertExpression ctx e
//        Expression.MemberInit(Expression.New(typeof<GroupKey>.GetConstructors().[0]),
//                              [ if e.Type = typeof<string> then
//                                    yield Expression.Bind(typeof<GroupKey>.GetProperty("String"), e) :> MemberBinding
//                                elif e.Type = typeof<int> then
//                                    yield Expression.Bind(typeof<GroupKey>.GetProperty("Number"), e) :> MemberBinding
//                                elif e.Type = typeof<float> then
//                                    yield Expression.Bind(typeof<GroupKey>.GetProperty("NumberF"), e) :> MemberBinding
//                                elif e.Type = typeof<DateTime> then
//                                    yield Expression.Bind(typeof<GroupKey>.GetProperty("Date"), e) :> MemberBinding
//                                else failwith "Not supported type of sort key" ]) :> Expression





module PagingQueryModule =
    open QueryModule

    let paging (query : #IQueryable<'a>) =
        function
        | Take(n) -> query.Take(n)
        | Skip(n) -> query.Skip(n)

[<AutoOpen>]
module UtilQuery =
    open QueryModule

    let columnInfo<'entity, 'a> (property : Expr<'entity -> 'a>) =
        let rec propertyNameGet property =
            match property with
            | PropertyGet(_, p, _) ->
                { ColumnName = p.Name
                  ColumnType = p.PropertyType }
            | Lambda(_, expr) -> propertyNameGet expr
            | _ -> failwith "Property name cannot be derived from the quotation passed to propName"
        propertyNameGet property

[<RequireQualifiedAccess>]
module Seq =
    open QueryModule

    let reduceWithOr lst =
        if lst |> Seq.isEmpty then Constant(Boolean(true))
        else lst |> Seq.reduce (fun p1 p2 -> Binary(p1, p2, Or))

    let reduceWithAnd lst =
        if lst |> Seq.isEmpty then Constant(Boolean(false))
        else lst |> Seq.reduce (fun p1 p2 -> Binary(p1, p2, And))


// ------------------------------------------------------------------------------------------------
// Translation of query transformations
// ------------------------------------------------------------------------------------------------
module QueryTransformatinoModule =
    open QueryFunctionModule
    open QueryModule
    open SortingModule
    open GroupingModule
    open PagingQueryModule
    open DataModel

    // Create LINQ lambda expression from our predicate - we create a variable `var`
    // of type `'entity`, convert the predicate into body expression and create
    // a lambda of type `Expression<Func<'entity, bool>>` that can be passed to `Where`
    let translatePredicate<'entity> (pred : FilterQuery) =
        let var = Expression.Parameter(typeof<'entity>)

        let ctx =
            { MethodGetter = methodGetter
              ColumnGetter = columnGetter<'entity> var }

        let body = convertExpression ctx pred
        Expression.Lambda<Func<'entity, bool>>(body, var)

    let translateSorting (sortBy : SortQuery) (q : IQueryable<'entity>) =
        match sortBy with
        | [] -> q
        | sortHead :: sortTail ->
            let parameter = Expression.Parameter(typeof<'entity>)
            let (e, ord) = sortHead

            let input =
                match ord with
                | Ascending -> q.OrderBy(makeFunction<'entity, SortKey> convertSortExpression e)
                | Descending -> q.OrderByDescending(makeFunction<'entity, SortKey> convertSortExpression e)

            let output = sortTail |> Seq.fold thenSortBy input
            upcast output

    let transform (input : IQueryable<'T>) tfs =
        match tfs with
        | Filter e ->
            // To transform filter, we just convert the expression to `Func<'T, bool>`
            // and call the LINQ `Where` operation
            input.Where(makeFunction<'T, bool> convertExpression e)
        | SortBy [] -> input
        | SortBy((e, ord) :: es) ->
            // Sorting is only a bit more complicated. We convert the *first* sorting condition
            // to `Func<'T, SortKey>` and then call OrderBy/OrderByDescending. For all the subsequent
            // operations, we call ThenBy/ThenByDescending using the helper function `thenSortBy`
            let input =
                match ord with
                | Ascending -> input.OrderBy(makeFunction<'T, SortKey> convertSortExpression e)
                | Descending -> input.OrderByDescending(makeFunction<'T, SortKey> convertSortExpression e)

            let output = es |> Seq.fold thenSortBy input
            upcast output
        | GroupBy(e, aggs) ->
            // Say we want to group players by name and count the total points per game.
            // We do this by generating LINQ tree that looks something like this - assuming `PPG` is
            // a value of type `IQueryable<PPG>`
            //
            //   palyers
            //     .GroupBy(m => new GroupKey { String = m.PlayerName })
            //     .Select(g => new PPG {
            //        PlayerName = g.Key,
            //        PPG = g.Select(m => m.ppg).Sum() });
            //
            // The type of `g` in the above is `IGrouping<GroupTuple, PPG>` and we create a variable for it:
            let var = Expression.Parameter(typeof<IGrouping<GroupTuple, 'T>>)

            // The aggregations are always of the form `g.Select(...).Agg()` except for the group key.
            // The group key is always available as `g.Key` and so we create expression for accessing it:
            let keyExpr = Expression.Property(var, "Key") :> Expression
            let keyTypes, groupFunc = makeGroupFunction<'T, GroupTuple> convertGroupExpression e
            let aggs = dict aggs

            // Grouping turns `IQueryable<GameEvent>` into another `IQueryable<GameEvent>`. To do this, we
            // need to return `new GameEvent { .. }` and set assignments for all the columns for which
            // we know how to aggregate them. (If we do not know, we just set them to a default value
            // using `getDefaultValue`.) So, get all the properties and get assignment for them:
            let aggs =
                [ for p in typeof<'T>.GetProperties() ->
                      let e =
                          if aggs.ContainsKey p.Name then
                              // Convert assignment!
                              convertAggregation<'T> var keyTypes keyExpr (aggs.[p.Name])
                          else
                              // Just return default value
                              upcast getDefaultValue p.PropertyType
                      Expression.Bind(p.SetMethod, e) :> MemberBinding ]

            // Turn this into LINQ function and call `GroupBy`
            let body = Expression.MemberInit(Expression.New(typeof<'T>.GetConstructors().[0]), aggs)
            let project = Expression.Lambda<Func<IGrouping<GroupTuple, 'T>, 'T>>(body, var)
            input.GroupBy(groupFunc).Select(project)

        | Paging(pgs) ->
            match pgs with
            | [] -> input
            | [ pg ] -> paging input pg
            | pg :: pgs -> (paging input pg, pgs) ||> Seq.fold paging

    let applyTransforms tfss input = tfss |> Seq.fold transform input


    let fromEntity =
      let v = Expression.Parameter(typeof<Medal>)
      Expression.Lambda<Func<Medal, MedalValue>>
        ( Expression.MemberInit
            ( Expression.New(typeof<MedalValue>.GetConstructors().[0]),
              [ for p in typeof<MedalValue>.GetProperties() do
                  yield Expression.Bind(p.SetMethod, Expression.Property(v, p.Name) ) :> MemberBinding ]), v )



    let shouldEqual (value : QueryValue) (colName : string) = Binary(GetColumn(colName), Constant(value), Equals)


module Tests =
    open QueryTransformatinoModule
    open QueryFunctionModule
    open QueryModule
    open SortingModule
    open GroupingModule
    open DataModel
    open System.Linq

    let db = new OlympicDbContext("Data Source=DESKTOP-0G0BUJ0;Initial Catalog=SimpleDB;Integrated Security=SSPI;")
    db.Database.Log <- fun s -> printfn "%s" s



    let transforms =
      [ // Get Italian medalists only!

        Filter( Binary(GetColumn "Team", Constant(QueryValue.String "ITA"), Equals))
        GroupBy
          ( [GetColumn "Athlete"; GetColumn "Year"],
            [ "Athlete", Aggregation.Key 0;
              "Year", Aggregation.Key 1;
              "Gold", Aggregation.Sum(GetColumn "Gold") ])
        SortBy [ GetColumn "Gold", Descending ] ]


    db.Medals.Select(fromEntity)
    |> applyTransforms transforms
    |> Seq.iter (fun a ->
      printfn "%s %d: %d gold medals" a.Athlete a.Year a.Gold)


    let transformsFilter = [

                Filter(Binary(
                            Binary(Binary(GetColumn "Sport", Constant(QueryValue.String "Aquatics"), Equals),
                                   Binary(GetColumn "Sport", Constant(QueryValue.String "Gymnastics"), Equals), Or),
                            Binary(GetColumn "Team", Constant(QueryValue.String "USA"), Equals), And))

                GroupBy
                      ( [
                            GetColumn "Athlete";
                            GetColumn "Sport";
                            GetColumn "Year"
                        ],

                        [
                         "Athlete", Aggregation.Key 0
                         "Sport", Aggregation.Key 1
                         "Year", Aggregation.Key 2

                        ])

                    // Get the athlete with most medals first
                SortBy [ GetColumn "Gold", Descending ]
            ]

    db.Medals.Select(fromEntity)
    |> applyTransforms transformsFilter
    |> Seq.iter (fun a -> printfn "%s %d: %d gold medals" a.Athlete a.Year a.Gold)
