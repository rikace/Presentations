namespace Easj360FSharp

open System
open System.Threading
open System.IO
open Microsoft.FSharp.Control


module AsyncCopy =
    
   type public RequestGate(n:int) =
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
        let gate = new RequestGate(Environment.ProcessorCount * 2)

        let parallel2 (job1, job2) =
            async { 
                    let! task1 = Async.StartChild job1
                    let! task2 = Async.StartChild job2
                    let! res1 = task1
                    let! res2 = task2
                    return (res1, res2) }

        let getStream (fileName:string, mode:FileMode) = 
                new FileStream(fileName, mode, FileAccess.ReadWrite, FileShare.ReadWrite, 4096, FileOptions.Asynchronous ||| FileOptions.SequentialScan)
       
        let getArray = Array.init 2 (fun _ -> Array.create 4096 0uy)
       

        member private x.ReadStream (stream:Stream, buffer:byte[]) = async {
                    return! stream.AsyncRead(buffer,0,buffer.Length)
                    }
       
        member private x.WriteStream (stream:Stream, buffer:byte[]) = async {
                    do! stream.AsyncWrite(buffer,0,buffer.Length)
                    return buffer.Length
                    }

        member private x.StartCopy(s:string, d:string) =  
            let source = getStream(s, FileMode.Open)
            let destination = getStream(d, FileMode.Create)
            let buffer = getArray
            let bufferIndex = ref 0
            Console.WriteLine(String.Format("Coping file {0}", Path.GetFileName(s)))
            let bytesRead = Async.RunSynchronously( x.ReadStream(source, buffer.[!bufferIndex]) )
            let rec copyAsync(index:int, bufIndex:int32 ref, data:byte[][]) = 
                async {
                    match index with
                    | 0  -> source.Close()
                            destination.Close()  
                            Console.WriteLine(String.Format("Coping {0} Completed", Path.GetFileName(s)))
                            return ()
                    | _  -> let write = x.WriteStream(destination, data.[!bufIndex].[0..index-1])
                            bufIndex := !bufIndex ^^^ 1                            
                            let read = x.ReadStream(source, buffer.[!bufferIndex])
                            let! op = parallel2(write, read)                            
                            return! copyAsync(int32(snd op), bufIndex,  data)
                    }
            copyAsync(int32(bytesRead), bufferIndex, buffer)
      
        member x.Copy(ds:string, dd:string) =
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
                                             do! x.StartCopy(f.FullName, combineFile(f))
                                           })                                            
               |> Async.Parallel
               |> Async.RunSynchronously
               |> ignore
