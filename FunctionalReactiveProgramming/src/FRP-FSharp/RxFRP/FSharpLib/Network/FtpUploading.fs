namespace Easj360FSharp
    open System
    open System.Threading
    open System.Net
    open System.IO
    open System.Xml
    open Microsoft.FSharp.Control

    module FtpUploading =
        let requestgate = AsyncGate.RequestGate(Environment.ProcessorCount)
 
        type System.Net.WebRequest with 
                member x.GetRequestStreamAsync() =
                    Async.FromBeginEnd(x.BeginGetRequestStream, x.EndGetRequestStream)

        type FtpUploadingData(ftpUrl:string, credential:System.Net.NetworkCredential) =    
            let uploadProcess(url:string, file:string) = async {
                use! gate = requestgate.Acquire()  
                let urlWfile = url.TrimEnd('/') + String.Format("/{0}", System.IO.Path.GetFileName(file))        
                printfn "processing %s to destination %s" file urlWfile                  
                let req = FtpWebRequest.Create(urlWfile) :?> FtpWebRequest
                req.Method <- System.Net.WebRequestMethods.Ftp.UploadFile        
                req.Credentials <- System.Net.NetworkCredential(credential.UserName, credential.Password)
                req.UseBinary <- true        
                req.KeepAlive <- false
                use! destination = req.GetRequestStreamAsync()
                let byteArray = Array.zeroCreate<byte>(4196)
                use! source = System.IO.File.AsyncOpenRead(file)
                try
                    let rec copy total =
                        async {
                                let! read = source.AsyncRead(byteArray,0, byteArray.Length)
                                if read = 0 then 
                                    return total
                                else
                                    do! destination.AsyncWrite(byteArray, 0, read)
                                    return! copy (total + int64(read))
                               }
                    return! copy 0L
                    finally
                        source.Close()
                        destination.Close()
                }
        
            member x.Start(files:seq<string>) =
                let tasks = Async.Parallel [ for file in files -> uploadProcess(ftpUrl, file) ]
                Async.RunSynchronously tasks
      
        
               
        
    
        
    