module Loader

open System
open System.IO
open FSharp.Data
open CommonTypes
open RethinkDb.Driver

let [<Literal>] IP = "192.168.99.100"
let [<Literal>] DataPath = "../Data/2015all.csv"
type Data = CsvProvider<DataPath, IgnoreErrors=true>
let isEquals (row:string) c = row.Equals("T", StringComparison.OrdinalIgnoreCase)


[<EntryPoint>]
let main argv =
     
    let data = Data.Load("../../../Data/2015all.csv")
    let port = RethinkDBConstants.DefaultPort
    let db = RethinkDB.R
    let conn = db.Connection().Hostname(IP).Port(port).Timeout(30).Connect()

    let tb = db.Db("baseball").Table("plays")
    // (1)  create the database and table in the RethinkDB manager dashboard
    conn.CheckOpen()

    for row in data.Rows |> Seq.take 10 do
        let play = 
            {   Play.id = Guid.NewGuid()
                gameId = row.GameId
                success = row.EventType
                homeTeam = row.GameId.Substring(0, 3)
                visitingTeam = row.VisitingTeam
                sequence = row.PitchSequence
                inning = row.Inning
                balls = row.Balls
                strikes = row.Strikes
                outs = row.Outs
                homeScore = row.HomeScore
                visitorScore = row.VisitorScore
                rbiOnPlay = row.RbiOnPlay
                hitValue = row.HitValue
                batter = row.BatterId
                pitcher = row.PItcherId
                isBatterEvent = isEquals row.IsBatterEvent "T"
                isAtBat = isEquals row.IsAtBat "T"
                isHomeAtBat = row.IsHomeAtBat
                isEndGame = isEquals row.IsEndGame "T"
                isSacFly = isEquals row.IsSacFly "T"
            }
        printf "Insert : %O" (tb.Insert(play).Run(conn))   
    printfn "Import completed"

    Console.WriteLine("Press any key to exit...")
    Console.ReadLine() |> ignore

    0
