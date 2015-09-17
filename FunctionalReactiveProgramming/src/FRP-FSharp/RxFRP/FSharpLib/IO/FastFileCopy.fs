namespace Easj360FSharp

open System
open System.Threading
open System.IO
open Microsoft.FSharp.Control

module FastParallel =

    let currentDir = __SOURCE_DIRECTORY__
    let currentFile = __SOURCE_FILE__

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
                        
    type FastParallelCopy (SOURCE:string) =    
         let gate = new RequestGate(8)
         let CopyFileAsync (fileNameIn:string, fileNameOut:string) = 
                        async {
                                 use! wait = gate.Acquire()
                                 use! inStream = File.AsyncOpen(fileNameIn, System.IO.FileMode.Open, System.IO.FileAccess.Read, System.IO.FileShare.Read, 4096, System.IO.FileOptions.Asynchronous ||| System.IO.FileOptions.SequentialScan)
                                 use! outStream = File.AsyncOpen(fileNameOut, System.IO.FileMode.Create, System.IO.FileAccess.Write, System.IO.FileShare.None, 4096, System.IO.FileOptions.Asynchronous ||| System.IO.FileOptions.SequentialScan)     
                                 let buffer = Array.zeroCreate<byte> 4096
                                 outStream.SetLength(inStream.Length)
                                 let rec copy total=
                                    async {
                                        let! read = inStream.AsyncRead(buffer,0,buffer.Length)
                                        if read=0 then 
                                            return total
                                        else
                                            do! outStream.AsyncWrite(buffer,0,read)
                                            return! copy (total + int64(read))
                                         }                                
                                 return! copy 0L
                                 }
                        
         let dir = new DirectoryInfo(SOURCE)
         let files = dir.GetFiles("*.*", SearchOption.AllDirectories)

         member x.Start() =         
                files
                |> Seq.map (fun x -> CopyFileAsync( x.FullName, Path.Combine(@"j:",x.Name) ) )
                |> Async.Parallel
                |> Async.RunSynchronously
                |> ignore                
                
    type FastCopy (SOURCE:string) =                        
        let CopyFileAsync (fileNameIn:string, fileNameOut:string) = 
                        async {
                                 use! inStream = File.AsyncOpen(fileNameIn, System.IO.FileMode.Open, System.IO.FileAccess.Read, System.IO.FileShare.Read, 4096, System.IO.FileOptions.Asynchronous ||| System.IO.FileOptions.SequentialScan)
                                 use! outStream = File.AsyncOpen(fileNameOut, System.IO.FileMode.Create, System.IO.FileAccess.Write, System.IO.FileShare.None, 4096, System.IO.FileOptions.Asynchronous ||| System.IO.FileOptions.SequentialScan)     
                                 let buffer = Array.zeroCreate<byte> 4096
                                 outStream.SetLength(inStream.Length)
                                 let rec copy total=
                                    async {
                                        let! read = inStream.AsyncRead(buffer,0,buffer.Length)
                                        if read=0 then 
                                            return total
                                        else
                                            do! outStream.AsyncWrite(buffer,0,read)
                                            return! copy (total + int64(read))
                                         }                                
                                return! copy 0L
                                 }

        let parallelWorker n f =
            MailboxProcessor.Start(fun inbox ->
                let workers = 
                    Array.init n (fun i -> MailboxProcessor.Start(f))
                let rec loop i = async {
                    let! msg = inbox.Receive()
                    workers.[i].Post(msg)
                    let incrI = (i + 1) % n
                    return! loop incrI
                }
                loop 0
            )

        let agent f =
            parallelWorker System.Environment.ProcessorCount (fun inbox ->
                let rec loop() = async {        
                    let! msg = inbox.Receive()
                    let s = (fst msg)
                    let d = (snd msg)
                    let! r = f (s,d)
                    r|> ignore
                    return! loop()
                }
                loop()
            )                        
                
        let dir = new DirectoryInfo(SOURCE)
        let agentCopy = agent CopyFileAsync
        let files = dir.GetFiles("*.*", SearchOption.AllDirectories)

        member x.Start() =         
                files
                |> Seq.iter (fun x -> agentCopy.Post( ( ( x.FullName) , (Path.Combine(@"f:\test",x.Name) ) ) ) )