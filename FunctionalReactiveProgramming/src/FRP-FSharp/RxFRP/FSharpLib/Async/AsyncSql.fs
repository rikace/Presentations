namespace Easj360FSharp

module public AsyncSql =

    open System.Data
    open System.Data.SqlClient  
    
    type internal System.Data.SqlClient.SqlCommand with
        member x.ExecuteAsyncReader() =
            Async.FromBeginEnd(x.BeginExecuteReader, x.EndExecuteReader)
        member x.ExecuteAsyncNonQuery() = 
            Async.FromBeginEnd(x.BeginExecuteNonQuery, x.EndExecuteNonQuery)
            
    let internal ExecuteInternalAsyncReader (command:System.Data.SqlClient.SqlCommand) =
        async {
                let! result = command.ExecuteAsyncReader()
                return result
              }
    
    let public ExecuteAsyncReader (command:System.Data.SqlClient.SqlCommand) =
            Async.RunSynchronously(ExecuteInternalAsyncReader command)
            
            
    type System.Data.SqlClient.SqlCommand with 
            member x.AsyncReader() =
                Async.FromBeginEnd(x.BeginExecuteReader, x.EndExecuteReader)
            member x.AsyncNonQuery() =        
                Async.FromBeginEnd(x.BeginExecuteNonQuery, x.EndExecuteNonQuery)
            
            
    type public AsyncSqlReader() =
        let _eventCompleted = new Event<System.ComponentModel.AsyncCompletedEventArgs>()
                
        let execReader(cmd:System.Data.SqlClient.SqlCommand) =
            async {
                        let! result = cmd.AsyncReader()
                        _eventCompleted.Trigger(System.ComponentModel.AsyncCompletedEventArgs(null,false, "Completed"))
                        return result
                  }
              
        let execNonQuery (cmd:System.Data.SqlClient.SqlCommand) =
            async {
                    try
                        let! result = cmd.AsyncNonQuery()                
                        return result
                     with 
                     |  :? System.Exception as e ->  return 0   
                  }
    
        member x.ExecuteAsyncReader(cmd:System.Data.SqlClient.SqlCommand) =
                Async.RunSynchronously(execReader cmd)
    
        member x.ExecuteAsyncNonQuery(cmd:System.Data.SqlClient.SqlCommand) =        
                Async.RunSynchronously(execNonQuery cmd)
                
        [<CLIEvent>]
        member x.EventCompleted = _eventCompleted.Publish   
            
            
    // type System.Data.SqlClient.SqlCommand with 
    //        member x.AsyncReader() =
    //            Async.BuildPrimitive(x.BeginExecuteReader, x.EndExecuteReader)
    //        member x.AsyncNonQuery() =        
    //            Async.BuildPrimitive(x.BeginExecuteNonQuery, x.EndExecuteNonQuery)
    //            
    //            
    //type public AsyncSql() =
    //    let _eventCompleted = new Event<System.ComponentModel.AsyncCompletedEventArgs>()
    //                
    //    let execReader(cmd:System.Data.SqlClient.SqlCommand) =
    //        async {
    //                    let! result = cmd.AsyncReader()
    //                    _eventCompleted.Trigger(System.ComponentModel.AsyncCompletedEventArgs(null,false, "Completed"))
    //                    return result
    //              }
    //              
    //    let execNonQuery (cmd:System.Data.SqlClient.SqlCommand) =
    //        async {
    //                let! result = cmd.AsyncNonQuery()                
    //                return result
    //              }
    //    
    //    member x.ExecuteAsyncReader(cmd:System.Data.SqlClient.SqlCommand) =
    //            Async.RunSynchronously(execReader cmd)
    //    
    //    member x.ExecuteAsyncNonQuery(cmd:System.Data.SqlClient.SqlCommand) =        
    //            Async.RunSynchronously(execNonQuery cmd)
    //                
    //    [<CLIEvent>]
    //    member x.EventCompleted = _eventCompleted.Publish      