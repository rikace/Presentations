module QueryModel

#r "../Lib/EntityFramework.dll"
#r "../Lib/EntityFramework.SqlServer.dll"
#r "System.Data.dll"
#r "System.Data.Linq.dll"
#r "System.ComponentModel.DataAnnotations.dll"
#r "System.Data.Entity.dll"
#load "Entities.fs"
#load "PredicateModel.fs"
#load "ExpUtils.fs"

open DataModel
open System
open System.Reflection
open System.Linq.Expressions
open System.Linq
open PredicateModel
open System.Data.Entity
open System.Reflection
open FSharp.Quotations
open FSharp.Quotations.Patterns

module QueryBuilder =
// Create LINQ lambda expression from our predicate - we create a variable `var`
// of type `Team`, convert the predicate into body expression and create
// a lambda of type `Expression<Func<Student, bool>>` that can be passed to `Where`
    open QueryModel
    open PredicateQueryModule

    let translate<'entity> pred =
        let typeEntity = typeof<'entity>
        let var = Expression.Parameter(typeEntity)
        let ctx =
            { MethodGetter = methodGetter
              ColumnGetter = columnGetter typeEntity var }
        let body = convertPredicate ctx pred
        Expression.Lambda<Func<'entity, bool>>(body, var)

module SortingBuilder =
    open QueryModel
    open SortQueryModule

    let transformSorting (sortBy:SortBy) (q:IQueryable<'a>)=
        match sortBy |> getSorting with
        | [] ->  q
        | sortHead::sortTail ->
            let parameter = Expression.Parameter(typeof<'a>)
            let orderedQuery = sortQuery q parameter sortHead
            (orderedQuery, sortTail)
            ||> List.fold (fun st cond -> sortOrderedQuery st parameter cond)
            :> IQueryable<'a>

module PagingBuilder =
    open QueryModel
    open PagingQueryModule

    let transformPaging (paging:Paging list) (q:IQueryable<'a>)=
        match paging with
        | [] -> q
        | [p] -> page p q
        | pg -> pg |> List.fold(fun q p -> page p q) q

module ExampleQueryPredicate =
    open QueryModel
    open PagingQueryModule
    open SortQueryModule
    open PredicateQueryModule
    open DataModel
    open System.Linq

    let db = new SimpleDbContext("Data Source=.;Initial Catalog=SimpleDB;Integrated Security=SSPI;")
    db.Database.Log <- fun s -> printfn "%s" s


    let students =
        db.Students
            .Where(fun s -> s.Name.Contains("Pe") && s.Age = Nullable(20) && s.Score > 1)
            .Select(fun s -> s.Name) |> List.ofSeq |> List.iter(printfn "%A")


    let age20 = And(Condition(Equals(GetColumn("Age"), Constant(NullNumber(Nullable 20)))),
                    Condition(Call(StringContains, [ GetColumn("Name"); Constant(String "Pe") ])))


    let agePred = QueryBuilder.translate<Student> age20
    db.Students.Where(agePred) |> Seq.toList


    let ageOver10 = RelationalBinary(GetColumn("Age"), Constant(NullNumber(Nullable 10)), GreaterThan) |> Condition

    let ageOver10Pred = QueryBuilder.translate<Student> ageOver10
    db.Students.Where(ageOver10Pred) |> Seq.toList


    let nameStartBy = Call(StringContains, [ GetColumn("Name"); Constant(String "Pe") ])

    let nameStartByName = QueryBuilder.translate<Student> (Condition(nameStartBy))
    db.Students.Where(nameStartByName) |> Seq.toList


    let nlst = System.Collections.Generic.List<Nullable<int>>()
    [1..20] |> List.iter(fun i -> nlst.Add(Nullable<int>(i)))
    nlst.Add(Nullable<int>())

    let seasonYears = Some([1..20])
    let teamId = [3..5]

    let pred =
        teamId
        |> List.map(fun id ->
                let pred = Condition(Equals(GetColumn("StudentID"), Constant(Number(id))))
                match seasonYears with
                | None -> pred
                | Some yearIds -> And(pred,  Condition(ContainsElements(InMemoryInts(yearIds),GetColumn("Age")))))
        |> List.reduce(fun p1 p2 -> Or(p1,p2))


    let predicate = QueryBuilder.translate<Student> pred
    db.Students.Where(predicate) |> Seq.toList


    type ColumnNane = string

    let shouldEqual (value : QueryValue) (colName : ColumnNane) = Condition(Equals(GetColumn(colName), Constant(value)))
    let nullInt v = (NullNumber(Nullable(v)))
    let Age = "Age"


    let predEQ = Age |> shouldEqual (nullInt 10)

    let predEq = QueryBuilder.translate<Student> predEQ
    db.Students.Where(predEq) |> Seq.toList




    let shouldBeGreaterEqualThan (value:int) (colName : ColumnNane) =
        Condition(RelationalBinary(GetColumn(colName), Constant(QueryValue.Number(value)), RelationOperator.GreaterThanOrEqual))

    let (&&&) a b = And(a,b)


    let ``age and score`` =  predEQ &&& ("Score" |> shouldBeGreaterEqualThan 1)

    let predEq' = QueryBuilder.translate<Student> ``age and score``
    db.Students.Where(predEq') |> Seq.toList
