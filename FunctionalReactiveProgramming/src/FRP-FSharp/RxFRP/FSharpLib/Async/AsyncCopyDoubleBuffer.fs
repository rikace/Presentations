namespace Easj360FSharp

module AsyncCopyDoubleBuffer =
    
    open System
    open System.IO
    open System.Threading
    open Microsoft.FSharp.Control
    open Microsoft.FSharp.Core
    open System.Security.Cryptography

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

      type public AsyncCopy() =    
        let gate = new RequestGate(4)
        member private x.getStream (fileName:string, mode:FileMode) = 
                new FileStream(fileName, mode, FileAccess.ReadWrite, FileShare.ReadWrite, 4096, FileOptions.Asynchronous ||| FileOptions.SequentialScan)
       
        member private x.getArray = Array.init 2 (fun _ -> Array.create 8192 0uy)
       
        member private x.ReadStream (stream:Stream, buffer:byte[]) = async {
                    return! stream.AsyncRead(buffer,0,buffer.Length)
                    }
       
        member private x.WriteStream (stream:Stream, buffer:byte[]) = async {
                    do! stream.AsyncWrite(buffer,0,buffer.Length)
                    return buffer.Length
                    }

        member private x.ComputeHash(s:Stream) =
            use hashAlg = new SHA1Managed()
            hashAlg.ComputeHash(s)                

        member private x.StartCopyFast(s:string, d:string) =  
            let source = x.getStream(s, FileMode.Open)
            let destination = x.getStream(d, FileMode.Create)
            let buffer = x.getArray
            let bufferIndex = ref 0
            Console.WriteLine(String.Format("Coping file {0}", Path.GetFileName(s)))
            let bytesRead = Async.RunSynchronously( x.ReadStream(source, buffer.[!bufferIndex]) )
            let sourceHash = x.ComputeHash(source)
            let rec copyAsync(index:int, bufIndex:int32 ref, data:byte[][]) = 
                async {
                try
                    match index with
                    | 0  -> source.Close()
                            destination.Close()  
                            let success =  x.CompareFiles(s,d) 
                            match success with
                                            | true ->  let originalColor = Console.ForegroundColor
                                                       Console.ForegroundColor <- ConsoleColor.Green
                                                       Console.WriteLine("Copied file {0} with SUCCESS",  Path.GetFileName(s))
                                                       Console.ForegroundColor <- originalColor   
                                            | false -> let originalColor = Console.ForegroundColor
                                                       Console.ForegroundColor <- ConsoleColor.Red
                                                       Console.WriteLine("Copied file {0} with UNSUCCESS",  Path.GetFileName(s))
                                                       Console.ForegroundColor <- originalColor           
                            return ()
                    | _  -> let write = x.WriteStream(destination, data.[!bufIndex].[0..index-1])
                            bufIndex := !bufIndex ^^^ 1
                            let read = x.ReadStream(source, buffer.[!bufferIndex])
                            let op = [write; read] 
                            let result = Async.RunSynchronously( Async.Parallel ( op ) )
                            return! copyAsync(int32(result.[1]), bufIndex,  data)
                with
                | _ ->  let originalColor = Console.ForegroundColor
                        Console.ForegroundColor <- ConsoleColor.Red
                        Console.WriteLine("Error Coping file {0}",  Path.GetFileName(s))
                        Console.ForegroundColor <- originalColor
                    }
            copyAsync(int32(bytesRead), bufferIndex, buffer)
        
        member private x.CompareFiles(aFrom:string, aTo:string) =  
            let f1 = new FileInfo(aFrom)
            let f2 = new FileInfo(aTo)
            if f1.Length = f2.Length then
                let seq1 = File.ReadAllBytes(f1.FullName)
                let seq2 = File.ReadAllBytes(f2.FullName)
                let compareSequences = Seq.compareWith (fun elem1 elem2 ->
                                                            if elem1 > elem2 then 1
                                                            elif elem1 < elem2 then -1
                                                            else 0)
                compareSequences seq1 seq2 = 0
            else
                false

        member private x.StartCopy(s:string, d:string,bufferSize:int) =  
            let source = x.getStream(s, FileMode.Open)
            let destination = x.getStream(d, FileMode.Create)
            let buffer = x.getArray
            let bufferIndex = ref 0
            Console.WriteLine(String.Format("Coping file {0}", Path.GetFileName(s)))           
            let buffer = Array.zeroCreate<byte> bufferSize
            let rec CopyAsyncRecStream (aFrom:FileStream, aTo:FileStream) =
                async {
                                    try
                                        let! read = aFrom.AsyncRead(buffer,0,buffer.Length)
                                        if read=0 then 
                                            aFrom.Flush()
                                            ignore(aFrom.Seek(0L, SeekOrigin.Begin))
                                            ignore(aTo.Seek(0L, SeekOrigin.Begin))                                           
                                            aFrom.Close()
                                            aTo.Close()                           
                                            let success =  x.CompareFiles(s,d)                                                 
                                            match success with
                                            | true ->  let originalColor = Console.ForegroundColor
                                                       Console.ForegroundColor <- ConsoleColor.Green
                                                       Console.WriteLine("Copied file {0} with SUCCESS",  Path.GetFileName(s))
                                                       Console.ForegroundColor <- originalColor   
                                            | false -> let originalColor = Console.ForegroundColor
                                                       Console.ForegroundColor <- ConsoleColor.Red
                                                       Console.WriteLine("Copied file {0} with UNSUCCESS",  Path.GetFileName(s))
                                                       Console.ForegroundColor <- originalColor                                            
                                        else
                                            do! aTo.AsyncWrite(buffer,0,read)
                                            return! CopyAsyncRecStream(aFrom, aTo)
                                     with 
                                    | ex -> aFrom.Close()
                                            aTo.Close()
                                            let originalColor = Console.ForegroundColor
                                            Console.ForegroundColor <- ConsoleColor.Red
                                            Console.WriteLine("Error Coping file {0}",  Path.GetFileName(s))
                                            Console.ForegroundColor <- originalColor
                                            return ()
                       }
            CopyAsyncRecStream(source, destination)
      
        member x.Copy(ds:string, dd:string, fast:bool) =
               let dir = new DirectoryInfo(ds)
               let files = dir.GetFiles("*.*")
               let checkAndCreateDir dir = if Directory.Exists(dir) = false 
                                                then ignore(Directory.CreateDirectory(dir))
               let combineFile (fName:FileInfo) = 
                    let destinationPath = fName.FullName.Replace(fName.Directory.Root.FullName, "")
                    let combine = Path.Combine(dd, destinationPath)
                    checkAndCreateDir(Path.GetDirectoryName(combine))
                    combine                
               files
               |> Array.map(fun f -> async { use! g = gate.Acquire()
                                             match fast with
                                             | true -> do! x.StartCopyFast(f.FullName, combineFile(f))
                                             | false -> do! x.StartCopy(f.FullName, combineFile(f), 4096*8)
                                           })                                            
               |> Async.Parallel
               |> Async.RunSynchronously
               |> ignore






        
        
