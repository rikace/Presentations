namespace Easj360FSharp

open System.Data
open System.Data.SqlClient

type SqlParam =
    { Name:string
      Value:System.Object }
   
type SqlCommandData =
    { sql: string
      parameters: SqlParam array
      cmdType: CommandType
      connectionString: string };;
      
module public SqlFetcher =
  open System.Data
  open System.Data.SqlClient
  open System.Configuration
  open System.Xml
  
  type internal System.Data.SqlClient.SqlCommand with
    member x.ExecuteReaderAsync() =
      Async.FromBeginEnd(x.BeginExecuteReader, x.EndExecuteReader)
    member x.ExecuteNonQueryAsync() =
      Async.FromBeginEnd(x.BeginExecuteNonQuery, x.EndExecuteNonQuery)

  let internal BuildCommand connection (data:SqlCommandData) = 
    let result = 
      new SqlCommand(data.sql, connection)
    let parameters = 
      data.parameters 
      |> Seq.map (fun p -> new SqlParameter(p.Name, p.Value)) 
      |> Seq.toArray
    result.CommandType <- data.cmdType
    result.Parameters.AddRange(parameters)
    result

  let internal ReadAndMapAsync data (premap:IDataReader -> unit) (mapper:IDataRecord -> 'a) =
    let mapReader (rdr:IDataReader) = 
      seq { while rdr.Read() do yield mapper rdr } // seq workflow
    async { // async workflow
            use connection = new SqlConnection (data.connectionString)
            connection.Open()
            use command = BuildCommand connection data
            let! rdr = command.ExecuteReaderAsync() 
            premap rdr
            let result = mapReader rdr
            return result |> Seq.toArray } // note: we need to avoid lazy calculation here... or the reader will have been disposed
            
  let internal ExecuteNonQueryAsync data =
    async {
      use connection = new SqlConnection (data.connectionString)
      use command = BuildCommand connection data
      let! result = command.ExecuteNonQueryAsync()
      return result }  
           
  
  // Async methods with postbacks
//  let ReadAndMapAsyncWithPostback data premap mapper postback = Async.SpawnThenPostBack(ReadAndMapAsync data premap mapper, postback)
//  let ExecuteNonQueryAsyncWithPostback data postback = Async.SpawnThenPostBack(ExecuteNonQueryAsync data, postback)

  let c = @"Data Source=TEKSERVER\SQLENT2008;Initial Catalog=AdventureWorks;User ID=sa;Password=Jocker74!"
   
  let pep conn =         
        async {
            use sqlConn = new SqlConnection(conn)
            use cmd = new SqlCommand("SELECT * FROM [AdventureWorks].[Person].[Address]", sqlConn)
            sqlConn.Open()
            use! reader = cmd.ExecuteReaderAsync()                
            let pep = async {
                                seq {
                                while reader.Read() do
                                    yield reader.GetString(0).ToString() // :?> string
                                }
                }
//            let res = Async.Start pep |> Seq.toList
//            res
            pep
            }
    

