namespace Easj360FSharp

open System
open System.Threading
open System.Net
open System.IO
open System.Xml
open Microsoft.FSharp.Control.WebExtensions

type FtpDownloadinggDataEvent(ftpUrl:string, credential:System.Net.NetworkCredential, destinationFolder:string) = 
    let requestgate = AsyncGate.RequestGate(Environment.ProcessorCount) 
    let _eventCompleted = new Event<System.ComponentModel.AsyncCompletedEventArgs>()
    let _eventProgres = new Event<System.ComponentModel.ProgressChangedEventArgs>()
    let donwloadProcess(url:string, filename:string, destFolder:string) = async {
        use! gate = requestgate.Acquire()  
        printfn "processing %s to destination %s" url filename
        let req = FtpWebRequest.Create(url) :?> FtpWebRequest
        req.Method <- System.Net.WebRequestMethods.Ftp.DownloadFile
        req.Credentials <- System.Net.NetworkCredential(credential.UserName, credential.Password)
        req.UseBinary <- true        
        req.KeepAlive <- false
        use! resp = req.AsyncGetResponse()
        use source = (resp :?> FtpWebResponse).GetResponseStream()        
        use destination = new System.IO.FileStream(System.IO.Path.Combine(destFolder, filename), System.IO.FileMode.Create, System.IO.FileAccess.Write,  System.IO.FileShare.None, 4196, true)
        let byteArray = Array.zeroCreate<byte>(4196)
        let totalSize = (int)resp.ContentLength
        try
            let rec copy total =
                async {
                        let! read = source.AsyncRead(byteArray,0, byteArray.Length)
                        if read = 0 then
                            _eventCompleted.Trigger(System.ComponentModel.AsyncCompletedEventArgs(null, false, "ciao"))
                            return total
                        else
                            do! destination.AsyncWrite(byteArray, 0, read)
                            let percente = ((int)destination.Position * 100) / totalSize
                            _eventProgres.Trigger(new System.ComponentModel.ProgressChangedEventArgs(percente, "Hello"))
                            return! copy (total + int64(read))
                       }
            return! copy 0L
            finally
                source.Close()
                destination.Close()
        }
        
       
    [<CLIEvent>]
    member x.EventCompleted = _eventCompleted.Publish
    
    [<CLIEvent>]
    member x.EventProgress = _eventProgres.Publish
    
    member x.Start(files:seq<string>) =
        let tasks = Async.Parallel [ for file in files ->
                                     donwloadProcess((ftpUrl + "/" + file.TrimEnd('\r')), file.Remove(file.IndexOf("/"), 1).TrimEnd('\r').Remove(0, file.LastIndexOf("/")), destinationFolder) ]
        Async.RunSynchronously tasks
      
        
               
module public AsyncSqlEvent =
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
            

//    type public AsyncSqlReader() =
//        let _eventCompleted = new Event<System.ComponentModel.AsyncCompletedEventArgs>()
//                
//        let execReader(cmd:System.Data.SqlClient.SqlCommand) =
//             async {
//                    let! result = cmd.AsyncReader()
//                    _eventCompleted.Trigger(System.ComponentModel.AsyncCompletedEventArgs(null,false, "Completed"))
//                    return result
//              }
//
//        let execNonQuery (cmd:System.Data.SqlClient.SqlCommand) =
//            async {
//                try
//                    let! result = cmd.AsyncNonQuery()                
//                    return result
//                 with 
//                 |  :? System.Exception as e ->  return 0   
//              }
//
//        [<CLIEvent>]
//        member x.EventCompleted = _eventCompleted.Publish   


    type System.Data.SqlClient.SqlCommand with 
        member x.AsyncReader() =
            Async.FromBeginEnd(x.BeginExecuteReader, x.EndExecuteReader)
        member x.AsyncNonQuery() =        
            Async.FromBeginEnd(x.BeginExecuteNonQuery, x.EndExecuteNonQuery)
//        member x.ExecuteAsyncReader(cmd:System.Data.SqlClient.SqlCommand) =
//            let exec = new AsyncSqlReader()
//            Async.RunSynchronously(exec.execReader cmd)


//        member x.ExecuteAsyncNonQuery(cmd:System.Data.SqlClient.SqlCommand) =        
//            Async.RunSynchronously(execNonQuery cmd)                      
    
    
  

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
    
        
    