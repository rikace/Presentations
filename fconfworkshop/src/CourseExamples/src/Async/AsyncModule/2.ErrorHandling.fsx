open System.IO
open System
open System.Net

   
// ===========================================
// Async Error Handling
// ===========================================


let httpasync url =
    async { let req =  WebRequest.Create(Uri url)
            use! resp = req.AsyncGetResponse()
            use stream = resp.GetResponseStream()
            use reader = new StreamReader(stream)
            let contents = reader.ReadToEnd()
            return contents.Length }
 
let sites = [   "http://www.bing.com"; 
                "http://www.google.com";                  
                "xyz"; 
                "http://www.yahoo.com"; 
                "http://www.microsoft.com"]
 
let htmlOfSites =
    sites
    |> Seq.map (httpasync)
    |> Async.Parallel
    |> Async.RunSynchronously


            
let htmlOfSitesErrorHandling =
    sites
    |> Seq.map (httpasync)
    |> Async.Parallel
    |> Async.Catch 
    |> Async.RunSynchronously
    |> function
        | Choice1Of2 result     -> printfn "Async operation completed: %A" result
        | Choice2Of2 (ex : exn) -> printfn "Exception thrown: %s" ex.Message




let htmlOfSitesErrorHandlingContinutaion(sites) =
    let asyncOp = 
        sites
        |> Seq.map (httpasync)
        |> Async.Parallel

    let continuation result = printfn "Async operation completed: %A" result
    let exceptionContinutaion (ex:exn) = printfn "Exception thrown: %s" ex.Message
    let cancellationContinuation (cancel:OperationCanceledException) = printfn "Async operation cancelled"

    Async.StartWithContinuations(asyncOp, 
            continuation,
            exceptionContinutaion,
            cancellationContinuation)

htmlOfSitesErrorHandlingContinutaion(sites)





