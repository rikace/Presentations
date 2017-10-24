open System.IO
open System
open System.Net

// ===========================================
// Async Cancellation Handling
// ===========================================

let getCancellationToken() = new System.Threading.CancellationTokenSource()

let httpasync url =
    async { let req =  WebRequest.Create(Uri url)
            use! resp = req.AsyncGetResponse()
            use stream = resp.GetResponseStream()
            use reader = new StreamReader(stream)
            let contents = reader.ReadToEnd()
            do! Async.Sleep 2000
            printfn "%s - %d" url  contents.Length }
 
let sites = [
    "http://www.bing.com"; 
    "http://www.google.com"; 
    "http://www.yahoo.com";
    "http://www.facebook.com"; 
    "http://www.youtube.com"; 
    "http://www.reddit.com"; 
    "http://www.digg.com"; 
    "http://www.twitter.com"; 
    "http://www.gmail.com"; 
    "http://www.docs.google.com"; 
    "http://www.maps.google.com"; 
    "http://www.microsoft.com"; 
    "http://www.netflix.com"; 
    "http://www.hulu.com"] 

let htmlOfSites =
    sites
    |> Seq.map (httpasync)
    |> Async.Parallel
    |> Async.Ignore

    
let cancellationToken = getCancellationToken()
Async.Start(htmlOfSites, cancellationToken=cancellationToken.Token)

cancellationToken.Cancel()



let cancellationToken' = getCancellationToken()

// Callback used when the operation is canceled
let cancelHandler (ex : OperationCanceledException) =
    printfn "The task has been canceled."


let tryCancelledAsyncOp = Async.TryCancelled(htmlOfSites, cancelHandler)

Async.Start(tryCancelledAsyncOp, cancellationToken=cancellationToken'.Token)
    
cancellationToken'.Cancel()




let continuation result = 
        printfn "Async operation completed: %A" result
let exceptionContinutaion (ex:exn) = 
        printfn "Exception thrown: %s" ex.Message
let cancellationContinuation (cancel:OperationCanceledException) = 
        printfn "Async operation cancelled"

let cancellationToken'' = getCancellationToken()

Async.StartWithContinuations(htmlOfSites, 
            continuation,
            exceptionContinutaion,
            cancellationContinuation, 
            cancellationToken''.Token)
    
cancellationToken''.Cancel()

