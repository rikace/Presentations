open System
open System.IO

type FileAsyncCopy() =
    static member AsyncCopy(source, destination) =
        let copyDel = new Func<_*_,_>(System.IO.File.Copy)
        Async.FromBeginEnd((source,destination), copyDel.BeginInvoke, copyDel.EndInvoke)

let copy source destination =
    async { do! FileAsyncCopy.AsyncCopy(source, destination) } |> Async.Start


// callback back 2 back
let processFileAsync (filePath : string) (processBytes : byte[] -> byte[]) =

    // This is the callback from when the AsyncWrite completes
    let asyncWriteCallback = new AsyncCallback(fun (iar : IAsyncResult) ->
            // Get state from the async result
            let writeStream = iar.AsyncState :?> FileStream
            
            // End the async write operation by calling EndWrite
            let bytesWritten = writeStream.EndWrite(iar)
            writeStream.Close()
            
            printfn "Finished processing file [%s]" (Path.GetFileName(writeStream.Name))
        )
    
    // This is the callback from when the AsyncRead completes
    let asyncReadCallback = new AsyncCallback(fun (iar : IAsyncResult) -> 
            // Get state from the async result
            let readStream, data = iar.AsyncState :?> (FileStream * byte[])
            
            // End the async read by calling EndRead
            let bytesRead = readStream.EndRead(iar)
            readStream.Close()
            
            // Process the result
            printfn "Processing file [%s], read [%d] bytes" (Path.GetFileName(readStream.Name)) bytesRead
                
            let updatedBytes = processBytes data
            
            let resultFile = new FileStream(readStream.Name + ".result",
                                           FileMode.Create)
            
            let _ = resultFile.BeginWrite(updatedBytes, 0, updatedBytes.Length, 
                                                   asyncWriteCallback, resultFile)
            ()
        )

    // Begin the async read, whose callback will begin the async write
    let fileStream = new FileStream(filePath, FileMode.Open, FileAccess.Read, 
                                    FileShare.Read, 0x1000, FileOptions.Asynchronous)

    let fileLength = int fileStream.Length
    let buffer = Array.zeroCreate fileLength

    // State passed into the async read
    let state = (fileStream, buffer)
    
    printfn "Processing file [%s]" (Path.GetFileName(filePath))
    let _ = fileStream.BeginRead(buffer, 0, buffer.Length, 
                                 asyncReadCallback, state)
    ()

// ~~~~~~~~~~~~~~~~~~   Test it

// Create a text file filled entirely with 'a's
let allABytes = Array.init 1024 (fun _ -> byte 'a')

// FileToProcess should read "aaaaaaaaaaaaaaa..."
File.WriteAllBytes(__SOURCE_DIRECTORY__ + "\FileToProcess.txt", allABytes)

// Now update the file's contents asynchronously
let processData fileContents =
    fileContents
    |> Array.mapi (fun idx (b : byte) -> b + byte (idx % 26))
    
// After the async operation completes, 
// FileToProcess.txt.results will read "abcdefghijklmopqrstuv..."
processFileAsync (__SOURCE_DIRECTORY__ + "\FileToProcess.txt") processData


// F# workflow example
let asyncProcessFile (filePath : string) (processBytes : byte[] -> byte[]) =
    async {        
        printfn "Processing file [%s]" (Path.GetFileName(filePath))        
        use fileStream = new FileStream(filePath, FileMode.Open)
        let bytesToRead = int fileStream.Length
        
        let! data = fileStream.AsyncRead(bytesToRead)
        
        printfn "Opened [%s], read [%d] bytes" (Path.GetFileName(filePath)) data.Length
        
        let data' = processBytes data
        
        use resultFile = new FileStream(filePath + ".results", FileMode.Create)
        do! resultFile.AsyncWrite(data', 0, data'.Length)
        
        printfn "Finished processing file [%s]" <| Path.GetFileName(filePath)
    } |> Async.Start

// ~~~~~~~~~~~~~~~~~~   Test it

// Create a text file filled entirely with 'a's
let allABytes2 = Array.init 1024 (fun _ -> byte 'a')

// FileToProcess should read "aaaaaaaaaaaaaaa..."
File.WriteAllBytes(__SOURCE_DIRECTORY__ + "\FileToProcess2.txt", allABytes)

// Now update the file's contents asynchronously
let processData2 fileContents =
    fileContents
    |> Array.mapi (fun idx (b : byte) -> b + byte (idx % 26))
    
// After the async operation completes, 
// FileToProcess.txt.results will read "abcdefghijklmopqrstuv..."
asyncProcessFile (__SOURCE_DIRECTORY__ + "\FileToProcess2.txt") processData
