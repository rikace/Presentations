module AdoAccess

open System.Configuration
open System.Collections.Generic
open System.Data
open System.Data.SqlClient
open System.Data.Common
open System
open Microsoft.FSharp
//open FSharp.PowerPack

/// create and open an SqlConnection object using the connection string found
/// in the configuration file for the given connection name
let openSQLConnection(connName:string) =
    let connSetting = ConfigurationManager.ConnectionStrings.[connName]
    let conn = new SqlConnection(connSetting.ConnectionString)
    conn.Open()
    conn

/// create and execute a read command for a connection using
/// the connection string found in the configuration file
/// for the given connection name
let openConnectionReader connName cmdString =
    let conn = openSQLConnection(connName)
    let cmd = conn.CreateCommand(CommandText=cmdString,
                                 CommandType = CommandType.Text)
    let reader = cmd.ExecuteReader(CommandBehavior.CloseConnection)
    reader

/// read a row from the data reader
let readOneRow (reader: #DbDataReader) =
    if reader.Read() then
        let dict = new Dictionary<string, obj>()
        for x in [ 0 .. (reader.FieldCount - 1) ] do
            dict.Add(reader.GetName(x), reader.[x])
        Some(dict)
    else
        None

///// execute a command using the Seq.generate
//let execCommand (connName: string) (cmdString: string) =
//    Seq.generate
//        // This function gets called to open a connection and create a reader
//        (fun () -> openConnectionReader connName cmdString)
//        // This function gets called to read a single item in
//        // the enumerable for a reader/connection pair
//        (fun reader -> readOneRow(reader))
//        (fun reader -> reader.Dispose())
//
///// open the contacts table
//let contactsTable =
//    execCommand
//        "MyConnection"
//        "select * from Person.Contact"
//        
///// print out the data retrieved from the database
//for row in contactsTable do
//    for col in row.Keys do
//        printfn "%s = %O" col (row.Item(col))
