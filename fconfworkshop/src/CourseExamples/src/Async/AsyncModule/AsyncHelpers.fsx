open System.IO
open System
open System.Net

module ``Async vs Sync`` = 

    let httpsync url =
        let req =  WebRequest.Create(Uri url)
        use resp = req.GetResponse()
        use stream = resp.GetResponseStream()
        use reader = new StreamReader(stream)
        let contents = reader.ReadToEnd()
        contents

    let httpasync url =
        async { let req =  WebRequest.Create(Uri url)
                use! resp = req.AsyncGetResponse()
                use stream = resp.GetResponseStream()
                use reader = new StreamReader(stream)
                let contents = reader.ReadToEnd()
                return contents }
 
    let sites = [
    "http://www.bing.com"; "http://www.google.com"; "http://www.yahoo.com";
                "http://www.facebook.com"; "http://www.youtube.com"; "http://www.reddit.com"; "http://www.digg.com";
                "http://www.twitter.com"; "http://www.gmail.com"; "http://www.docs.google.com"; "http://www.maps.google.com"; "http://www.microsoft.com"; "http://www.netflix.com"; "http://www.hulu.com"]
 
    #time "on"
    let htmlOfSitesSync =
        [for site in sites -> httpsync site]

    let htmlOfSites =
        sites
        |> Seq.map (httpasync)
        |> Async.Parallel
        |> Async.RunSynchronously
    
module ``Async Error Handling`` = 

    let httpasync url =
        async { let req =  WebRequest.Create(Uri url)
                use! resp = req.AsyncGetResponse()
                use stream = resp.GetResponseStream()
                use reader = new StreamReader(stream)
                let contents = reader.ReadToEnd()
                return contents.Length }
 
    let sites = ["http://www.bing.com"; "http://www.google.com"; "http://www.yahoo.com";
                "http://xyz"; "http://www.microsoft.com"; "http://www.netflix.com"; "http://www.hulu.com"]
 

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




    let htmlOfSitesErrorHandlingContinutaion =
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



module ``Async Cancellation Handling`` = 

    let getCancellationToken() = new System.Threading.CancellationTokenSource()

    let httpasync url =
        async { let req =  WebRequest.Create(Uri url)
                use! resp = req.AsyncGetResponse()
                use stream = resp.GetResponseStream()
                use reader = new StreamReader(stream)
                let contents = reader.ReadToEnd()
                do! Async.Sleep 2000
                printfn "%s - %d" url  contents.Length }
 
    let sites = ["http://www.bing.com"; "http://www.google.com"; "http://www.yahoo.com";
                "http://www.facebook.com"; "http://www.youtube.com"; "http://www.reddit.com"; "http://www.digg.com";
                "http://www.twitter.com"; "http://www.gmail.com"; "http://www.docs.google.com"; "http://www.maps.google.com";
                "http://www.microsoft.com"; "http://www.netflix.com"; "http://www.hulu.com"]
 

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

    // Async.CancelDefaultToken()                


    let continuation result = printfn "Async operation completed: %A" result
    let exceptionContinutaion (ex:exn) = printfn "Exception thrown: %s" ex.Message
    let cancellationContinuation (cancel:OperationCanceledException) = printfn "Async operation cancelled"

    let cancellationToken'' = getCancellationToken()

    Async.StartWithContinuations(htmlOfSites, 
                continuation,
                exceptionContinutaion,
                cancellationContinuation, 
                cancellationToken''.Token)
    
    cancellationToken''.Cancel()

module ``Async Helper`` = 


    type Microsoft.FSharp.Control.Async with
      static member StartDisposable(op:Async<unit>, (?cancelHandler:OperationCanceledException -> unit)) =
   
        let ct = new System.Threading.CancellationTokenSource()
   
        match cancelHandler with
        | None -> Async.Start(op, ct.Token)
        | Some(c) -> let computation = Async.TryCancelled(op, c)
                     Async.Start(computation, ct.Token)
        { new IDisposable with 
            member x.Dispose() = ct.Cancel() }



    type Microsoft.FSharp.Control.Async with
      static member StartContinuationDisposable(op:Async<'a>,   
                                (?continuation:'a -> unit),
                                (?exceptionContinutaion:exn -> unit),
                                (?cancellationContinuation:OperationCanceledException -> unit)) =

        let ct = new System.Threading.CancellationTokenSource()
        
        let continuation = defaultArg continuation (fun _ -> ())
        let exceptionContinutaion = defaultArg exceptionContinutaion (fun _ -> ())
        let cancellationContinuation = defaultArg cancellationContinuation (fun _ -> ())

        Async.StartWithContinuations(op,
                    continuation,
                    exceptionContinutaion,
                    cancellationContinuation,
                    ct.Token) 

        { new IDisposable with 
                member x.Dispose() = ct.Cancel() }


