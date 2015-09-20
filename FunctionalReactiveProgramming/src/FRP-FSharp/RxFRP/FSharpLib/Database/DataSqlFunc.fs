namespace Easj360FSharp
  open System.Data
  open System.Data.SqlClient

  type Param =
    { Name:string
      Value:System.Object }
   
  type CommandData =
    { sql: string
      parameters: Param array
      cmdType: CommandType
      connectionString: string };;
      
    module public Fetcher =
      open System.Data
      open System.Data.SqlClient
      open System.Configuration
      open System.Xml
  
      type internal System.Data.SqlClient.SqlCommand with
        member x.ExecuteReaderAsync() =
          Async.FromBeginEnd(x.BeginExecuteReader, x.EndExecuteReader)
        member x.ExecuteNonQueryAsync() =
          Async.FromBeginEnd(x.BeginExecuteNonQuery, x.EndExecuteNonQuery)
        member x.ExecuteXmlReaderAsync() =
          Async.FromBeginEnd(x.BeginExecuteXmlReader, x.EndExecuteXmlReader)

      let internal BuildCommand connection (data:CommandData) = 
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
          seq { while rdr.Read() do yield mapper rdr }  
        async {  
                use connection =
                  new SqlConnection (data.connectionString)
                connection.Open()
                use command = 
                  BuildCommand connection data
                let! rdr = 
                  command.ExecuteReaderAsync() 
                premap rdr
                let result = mapReader rdr
                return result |> Seq.toArray } 
            
      let internal GetXmlAsync data =
        async {
            use connection = new SqlConnection (data.connectionString)
            use command = BuildCommand connection data
            use! rdr = command.ExecuteXmlReaderAsync()  
            return rdr.ReadOuterXml() }

      let internal ExecuteNonQueryAsync data =
        async {
          use connection = new SqlConnection (data.connectionString)
          use command = BuildCommand connection data
          let! result = command.ExecuteNonQueryAsync()
          return result }  
           
      // Synchronous methods   
      let ReadAndMap data premap mapper = Async.RunSynchronously(ReadAndMapAsync data premap mapper)   
      let GetXml data = Async.RunSynchronously(GetXmlAsync data) 
      let ExecuteNonQuery data = Async.RunSynchronously(ExecuteNonQueryAsync data)
  
      // Async methods with postbacks
//      let ReadAndMapAsyncWithPostback data premap mapper postback = Async.SpawnThenPostBack(ReadAndMapAsync data premap mapper, postback)
//      let GetXmlAsyncWithPotback data postback = Async.SpawnThenPostBack(GetXmlAsync data, postback)
//      let ExecuteNonQueryAsyncWithPostback data postback = Async.SpawnThenPostBack(ExecuteNonQueryAsync data, postback)
