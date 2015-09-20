namespace Easj360FSharp

    open System
    open System.Threading
    open System.Net
    open System.IO
    open System.Xml
    open Microsoft.FSharp.Control.WebExtensions
 
     type FtpGetFileSync(ftpUrl:string, credential:System.Net.NetworkCredential) =    
        let getFileProcess(url:string) = async {
            let req = FtpWebRequest.Create(url) :?> FtpWebRequest
            req.Method <- System.Net.WebRequestMethods.Ftp.ListDirectory
            req.Credentials <- System.Net.NetworkCredential(credential.UserName, credential.Password)
            req.UseBinary <- false
            req.KeepAlive <- false
            let! resp = req.AsyncGetResponse()
            use source = (resp :?> FtpWebResponse).GetResponseStream()
            use reader = new System.IO.StreamReader(source)
            let result = reader.ReadToEnd() // .AsyncReadToEnd()
            return result
            }
 
        member x.Start() =        
            Async.RunSynchronously( getFileProcess(ftpUrl) ) |> string
      
        
               
        
    
        
        (*
         let requestgate = SyncGate.RequestGate(Environment.ProcessorCount)

     type FtpGetSize(ftpUrl:string, credential:System.Net.NetworkCredential) =    
        let getFileSizeProcess(file:string) = async {
            use! gate = requestgate.Acquire()  
            let req = FtpWebRequest.Create((ftpUrl + "//" + file)) :?> FtpWebRequest
            req.Method <- System.Net.WebRequestMethods.Ftp.GetFileSize
            req.Credentials <- System.Net.NetworkCredential(credential.UserName, credential.Password)
    //        req.UseBinary <- false
    //        req.KeepAlive <- false
            use! resp = req.AsyncGetResponse()
            let size = resp.ContentLength
            return  file, size
            }
 
        member x.Start(files:seq<string>) =
            let tasks = Async.Parallel [ for file in files -> 
                                                getFileSizeProcess file ]
            Async.RunSynchronously tasks
        *)