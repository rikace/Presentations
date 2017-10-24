open System
open Microsoft.FSharp.Data.TypeProviders
open FSharp.Data.Experimental
open FSharp.Data
open FSharpx.TypeProviders

[<Literal>]
let connectionString = @"Server=.\SqlExpress;Initial Catalog=AdventureWorks2008R2;Integrated Security=SSPI;"

[<Literal>]
let sqlQuery = "
SELECT TOP (@top) [ProductDescriptionID]
      ,[Description] ,[rowguid] ,[ModifiedDate]
  FROM [Production].[ProductDescription] WHERE [ModifiedDate] > @ModifiedDate"

[<Literal>]
let invokeSp = "
    EXEC HumanResources.uspUpdateEmployeePersonalInfo
        @BusinessEntityID, @NationalIDNumber, @BirthDate,
        @MaritalStatus, @Gender"

type QueryProductSync = SqlCommand<sqlQuery, connectionString>
type QueryProductAsRecords = SqlCommand<sqlQuery, connectionString, ResultType=ResultType.Records>
type QueryPersonInfoSingletoneAsRecords = SqlCommand<"SELECT * FROM dbo.ufnGetContactInformation(@PersonId)", connectionString,ResultType = ResultType.Records, SingleRow = true>
type UpdateEmplInfoCommand = SqlCommand<invokeSp, connectionString>
type QueryPersonInfoSingletoneTuples = SqlCommand<"SELECT PersonID, FirstName, LastName FROM dbo.ufnGetContactInformation(@PersonId)", connectionString, SingleRow=true>

type csvFile = CsvProvider<"TestCsv.csv">

[<EntryPoint>]
let main argv =

    let runEx = Choice1Of7

    match runEx with
    | Choice7Of2 ->
            let tuples = QueryProductSync().Execute(top = 7L, ModifiedDate = System.DateTime.Parse "2002-06-01")
            for id, description, guid, date in tuples do
                printfn "Product : id: %d | Description: %s | DateModified: %A" id description date

    | Choice7Of2 ->
            let records = QueryProductAsRecords().Execute(top = 7L, ModifiedDate = System.DateTime.Parse "2002-06-01")
            for record in records do
                printfn "Product : id: %d | Description: %s | DateModified: %A" record.ProductDescriptionID record.Description record.ModifiedDate

    | Choice7Of3 ->
            QueryProductAsRecords().AsyncExecute(top = 7L,  ModifiedDate = System.DateTime.Parse "2002-06-01")
                |> Async.RunSynchronously
                |> Seq.iter (fun x ->
                    printfn "Product : id: %d | Description: %s | DateModified: %A" x.ProductDescriptionID x.Description x.ModifiedDate)

    | Choice7Of4 ->
            let person = QueryPersonInfoSingletoneAsRecords().AsyncExecute(PersonId = 2) |> Async.RunSynchronously
            match person.FirstName, person.LastName with
                | Some first, Some last -> printfn "Person id: %i, name: %s %s" person.PersonID first last
                | _ -> printfn "What's your name %i?" person.PersonID

    | Choice7Of5 ->
            QueryPersonInfoSingletoneTuples().Execute(PersonId = 2)
            |> (function
                | id, Some first, Some last -> printfn "Person id: %i, name: %s %s" id first last
                | id, _, _ -> printfn "What's your name %i?" id )

    | Choice7Of6 ->
            let nonQuery = new UpdateEmplInfoCommand()
            ignore <| nonQuery.Execute(BusinessEntityID = 2, NationalIDNumber = "245797967",
                                                BirthDate = System.DateTime(1965, 09, 01), MaritalStatus = "S", Gender = "F")


    | _ ->
            let csv = csvFile.Load("TestCsv.csv")
            let firstRow = csv.Data |> Seq.head
            let firstName = firstRow.FirstName
            let age = firstRow.Age
            let date = firstRow.S
            for row in csv.Data do
                printfn "First Name %s - Last Name %s - Age %d" row.FirstName row.LastName row.Age


    Console.ReadLine()|>ignore
    0

