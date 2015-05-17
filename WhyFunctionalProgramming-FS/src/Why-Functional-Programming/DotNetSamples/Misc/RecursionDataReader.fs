module RecursionDataReader

open System 
open System.Data 
open System.Data.SqlClient 

let openConnection name = 
    let connection = new SqlConnection() 
    let connectionString = "data source=.\SqlExpress;initial catalog=" + name + ";integrated security=SSPI" 
    connection.ConnectionString <- connectionString 
    connection.Open() 
    connection 

let db = openConnection "AdventureWorks" 
let sql = "select top 5 ContactID, FirstName, LastName from Person.Contact order by FirstName" 

let createReader (connection : SqlConnection) sql = 
    let command = connection.CreateCommand() 
    command.CommandText <- sql 
    command.ExecuteReader() 

let showDataIter (reader : SqlDataReader) = 
    while reader.Read() do 
        let id = Convert.ToInt32(reader.["ContactID"]) 
        let fname = reader.["FirstName"].ToString() 
        let lname = reader.["LastName"].ToString()
        printfn "%d %s %s" id fname lname
    reader.Close() 

createReader db sql |> showDataIter

//////////////////
let showDataRec (reader : SqlDataReader) = 
    let rec showData (reader : SqlDataReader) = 
        match reader.Read() with 
        | true ->   let id = Convert.ToInt32(reader.["ContactID"]) 
                    let fname = reader.["FirstName"].ToString() 
                    let lname = reader.["LastName"].ToString()
                    printfn "%d %s %s" id fname lname 
                    showData reader 
        | false -> reader.Close() 
    showData reader 

createReader db sql |> showDataRec
//////////////////

type Person = { PersonID: int; FirstName: string; LastName: string; }

let createPersonListIter (reader : SqlDataReader) = 
    let mutable list = [] 
    while reader.Read() do 
        let id = Convert.ToInt32(reader.["ContactID"]) 
        let fname = reader.["FirstName"].ToString() 
        let lname = reader.["LastName"].ToString() 
        let person = {PersonID = id; FirstName = fname; LastName = lname} 
        list <- person :: list 
    reader.Close() 
    list 

let people = createReader db sql |> createPersonListIter

//////////////////

let createPersonListRec (reader : SqlDataReader) = 
    let list = [] 
    let rec getData (reader : SqlDataReader) list = 
        match reader.Read() with 
        | true ->   let id = Convert.ToInt32(reader.["ContactID"]) 
                    let fname = reader.["FirstName"].ToString() 
                    let lname = reader.["LastName"].ToString() 
                    let person = {PersonID = id; FirstName = fname; LastName = lname} 
                    getData reader (person :: list) 
        | false -> reader.Close() 
                   list  

    getData reader list 

let people' = createReader db sql |> createPersonListRec

///////////////////////
let selectNames = createReader db sql 
                    |> createPersonListRec 
                    |> List.filter (fun r -> r.LastName.StartsWith("W")) 
                    |> List.map (fun r -> r.LastName + ", " + r.FirstName) 
                    |> List.sort 
                    |> Seq.distinct 
                    |> List.ofSeq


