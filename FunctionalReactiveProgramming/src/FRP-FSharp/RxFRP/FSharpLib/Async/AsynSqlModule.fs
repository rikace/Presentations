
module AsynSqlModule

open System.Data
open System.Data.SqlClient

type internal System.Data.SqlClient.SqlCommand with
    member x.ExecuteReaderAsync() =
      Async.FromBeginEnd(x.BeginExecuteReader, x.EndExecuteReader)
    member x.ExecuteNonQueryAsync() =
      Async.FromBeginEnd(x.BeginExecuteNonQuery, x.EndExecuteNonQuery)
    member x.ExecuteReaderXmlAsync() =
      Async.FromBeginEnd(x.BeginExecuteXmlReader, x.EndExecuteXmlReader)
    member x.ExecuteScalarAsync<'a>() =
        async {
            let result = x.ExecuteScalar()
            let castResult = ((box result) :?> 'a)
            return castResult
            }
        
type public AsyncSqlCommand (command:System.Data.SqlClient.SqlCommand) =
    let executeNonQueryAsync =
            async {
                return! command.ExecuteNonQueryAsync()
                }
                
    let executeRedaer =
        async {
            return! command.ExecuteReaderAsync()
            }
        
    let execureReaderXml =
        async {
            return! command.ExecuteReaderXmlAsync()
            }            
              
    member x.AsyncNonQuery() =
        Async.RunSynchronously(executeNonQueryAsync)
        
    member x.AsyncReader() =
        Async.RunSynchronously(executeRedaer)
        
    member x.AsyncReaderXml() =
        Async.RunSynchronously(execureReaderXml)        

    member x.AsyncScalar<'a>() =
        Async.RunSynchronously(command.ExecuteScalarAsync<'a>())