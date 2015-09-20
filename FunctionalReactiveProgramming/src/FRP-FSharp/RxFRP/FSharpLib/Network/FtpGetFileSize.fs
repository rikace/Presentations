namespace Easj360FSharp

open System
open System.Threading
open System.Net
open System.IO
open System.Xml
open Microsoft.FSharp.Control
open Microsoft.FSharp.Control.WebExtensions

type FtpGetSize(ftpUrl:string, credential:System.Net.NetworkCredential) =    
    let requestgate = AsyncGate.RequestGate(Environment.ProcessorCount)
    let getFileSizeProcess(file:string) = async {
        use! gate = requestgate.Acquire()  
        let req = FtpWebRequest.Create((ftpUrl + "//" + file)) :?> FtpWebRequest
        req.Method <- System.Net.WebRequestMethods.Ftp.GetFileSize
        req.Credentials <- System.Net.NetworkCredential(credential.UserName, credential.Password)
//      req.UseBinary <- false
//      req.KeepAlive <- false
        use! resp = req.AsyncGetResponse()
        let size = resp.ContentLength
        return  file, size
        }
 
    member x.Start(files:seq<string>) =
        let tasks = Async.Parallel [ for file in files -> getFileSizeProcess file ]
        Async.RunSynchronously tasks
        
        
        
