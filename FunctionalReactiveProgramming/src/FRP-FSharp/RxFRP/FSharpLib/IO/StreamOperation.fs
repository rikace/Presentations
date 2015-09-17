#light 

module StreamOperation

open System.IO
open Microsoft.FSharp.Control

let streamCopy bufferSize a b =
  let buffer = Array.zeroCreate bufferSize
  let rec copy (a:Stream) (b:Stream) =
    match a.Read(buffer, 0, bufferSize) with
    | 0 -> ()
    | readSize -> 
      b.Write(buffer, 0, readSize)
      copy a b
  copy a b


let Read(fileName:string) = 
                let read = async {                         
                            use! stream = System.IO.File.AsyncOpenRead(fileName)
                            let data = Array.create (int stream.Length) 0uy 
                            let!  bytesRead = stream.AsyncRead(data, 0, data.Length)                         
                            return data
                                 }
                Async.RunSynchronously(read)
                
let CopySync(fileIn:string, fileOut:string, bufferSize:int)=
        use streamIn = new System.IO.FileStream(fileIn, FileMode.Open, FileAccess.Read, FileShare.Read)
        use streamOut = new System.IO.FileStream(fileOut, FileMode.Create, FileAccess.Write, FileShare.None)
        let data = Array.create bufferSize 0uy 
        let mutable bytesRead = streamIn.Read(data, 0, data.Length)
        while bytesRead <> 0 do
            streamOut.Write(data, 0, bytesRead)
            bytesRead <- streamIn.Read(data, 0, data.Length)                

let streamMatchCopy bufferSize a b =
  let buffer = Array.zeroCreate bufferSize
  let rec copy (a:Stream) (b:Stream) =
    match a.Read(buffer, 0, bufferSize) with
    | 0 -> ()
    | readSize -> 
      b.Write(buffer, 0, readSize)
      copy a b
  copy a b

let CopySyncRec (aFrom:Stream) (aTo:Stream) (buferSize:int) =
    let buffer = Array.zeroCreate<byte> buferSize
    let mutable totalbytes = 0L
    let mutable repeat = true
    while repeat do
        let read = aFrom.Read(buffer,0,buffer.Length)
        if read=0 then
            repeat <- false
        else
            aTo.Write(buffer,0,read)
            totalbytes <- totalbytes + int64(read)
    done
    totalbytes
    
let CopySyncRecImmutable (aFrom:Stream) (aTo:Stream) (bufferSize:int) =
    let buffer = Array.zeroCreate<byte> bufferSize
    let rec copy total=
        let read = aFrom.Read(buffer,0,buffer.Length)
        if read=0 then
            total
        else
            aTo.Write(buffer,0,read)
            copy (total + int64(read))
    copy 0L
    
let CopyAsyncRecStream (aFrom:Stream) (aTo:Stream) (bufferSize:int) =
    async {
            let buffer = Array.zeroCreate<byte> bufferSize
            let rec copy total=
                async {
                        let! read = aFrom.AsyncRead(buffer,0,buffer.Length)
                        if read=0 then 
                            return total
                        else
                            do! aTo.AsyncWrite(buffer,0,read)
                            return! copy (total + int64(read))
                      }
            aFrom.Close()
            aTo.Close()
            return! copy 0L
          }
          
let CopyFileAsync (fileNameIn:string, fileNameOut:string, sizeBuffer:int) = async {
                             use! inStream = File.AsyncOpen(fileNameIn, System.IO.FileMode.Open, System.IO.FileAccess.Read, System.IO.FileShare.Read, sizeBuffer, System.IO.FileOptions.Asynchronous)
                             use! outStream = File.AsyncOpen(fileNameOut,System.IO.FileMode.Create, System.IO.FileAccess.Write, System.IO.FileShare.None, sizeBuffer, System.IO.FileOptions.Asynchronous)         
                             return! CopyAsyncRecStream inStream outStream sizeBuffer
                             }
                             
let CopyAsyncRec (fileNameIn:string, fileNameOut:string) (sizeBuffer:int) =
    async {
            use! aFrom = File.AsyncOpen(fileNameIn, System.IO.FileMode.Open, System.IO.FileAccess.Read, System.IO.FileShare.Read, sizeBuffer, System.IO.FileOptions.Asynchronous)
            use! aTo = File.AsyncOpen(fileNameOut,System.IO.FileMode.Create, System.IO.FileAccess.Write, System.IO.FileShare.None, sizeBuffer, System.IO.FileOptions.Asynchronous) 
            let buffer = Array.zeroCreate<byte> sizeBuffer 
            let rec copy total=
                async {
                        let! read = aFrom.AsyncRead(buffer,0,buffer.Length)
                        if read=0 then 
                            return total
                        else
                            do! aTo.AsyncWrite(buffer,0,read)
                            return! copy (total + int64(read))
                      }
            aFrom.Close()
            aTo.Close()
            return! copy 0L
          }                             
      
let ParallelCopy (files:System.Collections.Generic.List<string>, destination:string) =
        let tasks = [| for s in files -> CopyFileAsync(s, System.IO.Path.Combine(destination, System.IO.Path.GetFileName(s)), 4092) |]
        let taskParallel = Async.Parallel ( tasks )
        Async.RunSynchronously ( taskParallel )
        
let  openFile(fileName:string) =     
         async { use  fs = new  FileStream(fileName, FileMode.Open, FileAccess.Read, FileShare.Read, 8192, true)             
                 let  data = Array.create (int fs.Length) 0uy                       
                 let! bytesRead = fs.AsyncRead(data, 0, data.Length)             
                 do  printfn "Read Bytes: %i" bytesRead 
                }

let ReadFiles (path:string) =    
     let filePaths = Directory.GetFiles(path)
     let tasks = [ for filePath in filePaths -> openFile filePath ]     
     Async.RunSynchronously (Async.Parallel tasks)

// Takes a Stream, a buffer, an offset and a count and returns a value of type Async<int>
// stream was the only type that had to be specified; the others could be inferred.
let readAsync (stream : Stream) buffer offset count = 
    Async.FromBeginEnd((fun (callback,state) -> stream.BeginRead(buffer,offset,count,callback,state)),(fun (asyncResult) -> stream.EndRead(asyncResult)))


let readTextFile path = 
    async {
        use file = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read,1024, FileOptions.Asynchronous)
        use ms = new MemoryStream()
        let keepGoing = ref true
        let buffer = Array.create 4096 (byte 0)
        while !keepGoing do
            let! bytesRead = readAsync file buffer 0 4096
            if bytesRead = 0 then keepGoing := false
            ms.Write(buffer,0,bytesRead)
        return System.Text.Encoding.UTF8.GetString(ms.ToArray())
    }

