module CancellationExample

open System.Threading
open System.Net
open System.IO

let getWebPage (url:string) =
    async {
        let req = WebRequest.Create url
        let! resp = req.AsyncGetResponse()
        let stream = resp.GetResponseStream()
        let reader = new StreamReader(stream)
        do! Async.Sleep(2000)
        let! html = Async.AwaitTask (reader.ReadToEndAsync())
        printfn "%s" html
        return () }

let capability = new CancellationTokenSource()
let tasks =
    Async.Parallel [ getWebPage "http://www.google.com"
                     getWebPage "http://www.bing.com" ]
    |> Async.Ignore

Async.Start (tasks, cancellationToken=capability.Token)

capability.Cancel()