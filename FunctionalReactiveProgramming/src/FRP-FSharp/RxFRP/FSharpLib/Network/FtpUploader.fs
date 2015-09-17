namespace Easj360FSharp

open System
open System.IO
open System.Net

module FtpUploader =

        type RequestGate(n:int) =
            let sem = new System.Threading.Semaphore(n, n)
            member x.Acquire(?timeout) =
                async { let! ok = Async.AwaitWaitHandle(sem, ?millisecondsTimeout=timeout)
                    if (ok) then
                        return
                            { new System.IDisposable with
                                member x.Dispose() =
                                    sem.Release() |> ignore }
                    else
                        return! failwith "Couldn't acquire Gate" }  

        type Microsoft.FSharp.Control.Async with
                 static member Parallel2(actionOne, actionTwo) =
                    async { 
                            let! resOne = Async.StartChild actionOne
                            let! resTwo = Async.StartChild actionTwo
                            let! completeOne = resOne
                            let! completeTwo = resTwo
                            return completeTwo
                          }
                                                      
        type UploadData(url:string, credentials:NetworkCredential, concurrency:int) =  
            let urlAddress = url
            let credentials = credentials            
            let gate = new RequestGate(concurrency)
        

            member private x.Start(file:System.IO.FileInfo) =
                    async {
                        use! wait = gate.Acquire()                 
                        let req = FtpWebRequest.Create(urlAddress+"//"+file.Name) :?> FtpWebRequest
                        req.Method <- System.Net.WebRequestMethods.Ftp.UploadFile
                        req.Credentials <- credentials
                        req.UseBinary <- true
                        req.KeepAlive <- false
                        use inStream = new FileStream(file.FullName, System.IO.FileMode.Open, System.IO.FileAccess.Read, System.IO.FileShare.Read, 4096, System.IO.FileOptions.Asynchronous ||| System.IO.FileOptions.SequentialScan)  
                        use! outStream = req.AsyncGetRequestStream()                   
                        let sizeBuffer = 4096
                        let buffers = Array.init 2 (fun _ -> Array.zeroCreate<byte> sizeBuffer) 
                        let indexBuffer = 0   
                        let! bytesReadFirstRound = inStream.AsyncRead(buffers.[indexBuffer],0,sizeBuffer)
                        let rec copyAsync(bytesRead:int, arrayIndex:int) = async {
                            match bytesRead with
                            | 0 -> return ()
                            | m -> let readArrayIndex = arrayIndex
                                   let writeIndex = arrayIndex ^^^ 1
                                   let! bytesRead = Async.Parallel2(outStream.AsyncWrite(buffers.[readArrayIndex], 0, m), inStream.AsyncRead(buffers.[writeIndex],0,sizeBuffer))
                                   return! copyAsync(bytesRead, writeIndex)
                                   }
                        do! copyAsync(bytesReadFirstRound, indexBuffer)    
                    }

            member x.Upload(dataName:string) =                
                if File.Exists(dataName) then 
                    Async.Start( x.Start(new FileInfo(dataName)) )
                else if Directory.Exists(dataName) then 
                    let dir = new DirectoryInfo(dataName)
                    let files = dir.EnumerateFiles("*.*", SearchOption.TopDirectoryOnly)
                    files
                    |> Seq.map (fun f -> x.Start(f))
                    |> Async.Parallel
                    |> Async.RunSynchronously
                    |> ignore
                     



           
                   