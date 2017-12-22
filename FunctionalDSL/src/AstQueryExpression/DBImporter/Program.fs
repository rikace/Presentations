open System
open FSharp.Data
open System.IO

open System.Data
open System.Data.Linq
open Microsoft.FSharp.Data.TypeProviders
open Microsoft.FSharp.Linq

let [<Literal>] csvFile = @"C:\temp\FilterExpressionPoc\FilterExpressionPOC\GroupingPOC\medals.csv"
let [<Literal>] conn = @"Data Source=DESKTOP-195V044;Initial Catalog=SimpleDB;Integrated Security=SSPI;"

type Medals = CsvProvider<csvFile>
type MedalTable = SqlDataConnection<conn>

[<EntryPoint>]
let main argv =

    let medals = Medals.GetSample()
    let db = MedalTable.GetDataContext()

    // db.DataContext.Log <- System.Console.Out

    let data =
            medals.Rows
            |> Seq.take 1000
            |> Seq.mapi(fun i row ->
                MedalTable.ServiceTypes.Medals(
                    //ID = i + 1,
                    Games = row.Games,
                    Year = row.Year,
                    Sport = row.Sport,
                    Discipline = row.Discipline,
                    Athlete = row.Athlete,
                    Team = row.Team,
                    Gender= row.Gender,
                    Event = row.Event,
                    Metal = row.Metal,
                    Gold = Convert.ToInt32(row.Gold),
                    Silver = Convert.ToInt32(row.Silver),
                    Bronze = Convert.ToInt32(row.Bronze) ))
            |> Seq.toList



    try
        db.Medals.InsertAllOnSubmit(data)
        db.DataContext.SubmitChanges()
    with
    | ex -> let m = ex.Message
            printfn "%s" m







    0 // return an integer exit code
