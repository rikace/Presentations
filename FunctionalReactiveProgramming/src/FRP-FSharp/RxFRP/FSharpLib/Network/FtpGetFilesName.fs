namespace Easj360FSharp

open System
open System.Threading
open System.Net
open System.IO
open System.Xml
open Microsoft.FSharp.Control.WebExtensions
 
 type FtpGetFile(ftpUrl:string, credential:System.Net.NetworkCredential) =    
    let requestgate = AsyncGate.RequestGate(Environment.ProcessorCount)
    let getFileProcess(url:string) = async {        
        use! gate = requestgate.Acquire()  
        let req = FtpWebRequest.Create(url) :?> FtpWebRequest
        req.Method <- System.Net.WebRequestMethods.Ftp.ListDirectory
        req.Credentials <- System.Net.NetworkCredential(credential.UserName, credential.Password)
        req.UseBinary <- false
        req.KeepAlive <- false
        let! resp = req.AsyncGetResponse()
        use source = (resp :?> FtpWebResponse).GetResponseStream()
        use reader = new System.IO.StreamReader(source)
        let! result = reader.AsyncReadToEnd()
        return result
        }
 
    member x.Start() =        
        Async.RunSynchronously( getFileProcess(ftpUrl) ) |> string
      
        
               
        
    
        
    