namespace Easj360FSharp

module FtpDownloading =

    open System
    open System.Threading
    open System.Net
    open System.IO
    open System.Xml
    open Microsoft.FSharp.Control

    let requestgate = AsyncGate.RequestGate(Environment.ProcessorCount) 
 
    type FtpDownloadinggData(ftpUrl:string, credential:System.Net.NetworkCredential, destinationFolder:string) = 
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
      
        
               
        
    
        
    