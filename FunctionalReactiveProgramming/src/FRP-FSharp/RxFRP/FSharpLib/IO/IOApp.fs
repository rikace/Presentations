namespace Easj360FSharp 

open System.IO
open System.Net

module Operation = 

    let AsyncFetch (url:string) = async {
        let req = WebRequest.Create(url)
        use! resp = req.AsyncGetResponse()
        use reader = new StreamReader(resp.GetResponseStream())
        let! html = reader.AsyncReadToEnd()
        //do printfn "Read %d chars from %s" html.Length url 
        return html
        }
  
    type System.Net.WebRequest with 
            member x.GetResponseAsync() =
                Async.FromBeginEnd(x.BeginGetResponse, x.EndGetResponse)
                
    let  openFile(fileName:string) =     
             async { use  fs = new  FileStream(fileName, FileMode.Open, FileAccess.Read, FileShare.Read, 8192, true)             
                     let  data = Array.create (int fs.Length) 0uy                       
                     let! bytesRead = fs.AsyncRead(data, 0, data.Length)             
                     do  printfn "Read Bytes: %i" bytesRead 
                    }

    //let result(fileName:string) = 
    //    try
    //        Async.RunSynchronously (openFile fileName)
    //    with 
    //        | :? System.OperationCanceledException -> async{}

    let ReadFiles (path:string) =    
         let filePaths = Directory.GetFiles(path)
         let tasks = [ for filePath in filePaths -> openFile filePath ]     
         Async.RunSynchronously (Async.Parallel tasks)

    let rec fib x =
        match x with
        | 1 -> 1
        | 2 -> 1
        | x -> fib (x - 1) + fib (x - 2)
    
    let  openFile2(fileName:string) =     
             async { use  fs = new  FileStream(fileName, FileMode.Open, FileAccess.Read, FileShare.Read)             
                     let  data = Array.create (int fs.Length) 0uy             
                     let! bytesRead = fs.AsyncRead(data, 0, data.Length)             
                     do  printfn "Read Bytes: %i" bytesRead 
                    }
         
                

    let result2(fileName:string) = 
        Async.RunSynchronously (openFile fileName)
        System.Console.ReadLine() 



         