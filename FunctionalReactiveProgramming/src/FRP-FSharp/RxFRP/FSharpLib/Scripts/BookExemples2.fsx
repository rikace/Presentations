#load "AgentSystem.fs"
#r "FSharp.PowerPack.dll"
open AgentSystem.LAgent

open System.Collections.Generic
open System.Net
open System.IO
open System.Threading
open System.Text.RegularExpressions
open Microsoft.FSharp.Control
open Microsoft.FSharp.Control.CommonExtensions
//open Microsoft.FSharp.Control.WebExtensions

let limit = 50
let linkPat = "href=\s*\"[^\"h]*(http://[^&\"]*)\""
let getLinks (txt:string) =
    [ for m in Regex.Matches(txt,linkPat)  -> m.Groups.Item(1).Value ]

// A type that helps limit the number of active web requests
type RequestGate(n:int) =
    let semaphore = new Semaphore(initialCount=n,maximumCount=n)
    member x.AcquireAsync(?timeout) =
        async { let! ok = Async.AwaitWaitHandle(semaphore, ?millisecondsTimeout=timeout)
                if ok then
                   return
                     { new System.IDisposable with
                         member x.Dispose() =
                             semaphore.Release() |> ignore }
                else
                   return! failwith "couldn't acquire a semaphore" }

// Gate the number of active web requests
let webRequestGate = RequestGate(5)

// Fetch the URL, and post the results to the urlCollector.
let collectLinks (url:string) =
    async { // An Async web request with a global gate
            let! html =
                async { // Acquire an entry in the webRequestGate. Release
                        // it when 'holder' goes out of scope
                        use! holder = webRequestGate.AcquireAsync()

                        // Wait for the WebResponse
                        let req = WebRequest.Create(url,Timeout=5)

                        use! response = req.AsyncGetResponse()

                        // Get the response stream
                        use reader = new StreamReader(response.GetResponseStream())

                        // Read the response stream
                        return! reader.AsyncReadToEnd()  }

            // Compute the links, synchronously
            let links = getLinks html

            // Report, synchronously
            do printfn "finished reading %s, got %d links" url (List.length links)

            // We're done
            return links }

//let urlCollector =
//    MailboxProcessor.Start(fun self ->
//        let rec waitForUrl (visited : Set<string>) =
//           async {
//                   if visited.Count < limit then
//                       let! url = self.Receive()
//                       if not (visited.Contains(url)) then
//                           do! Async.SpawnChild
//                                   (async { let! links = collectLinks url
//                                            for link in links do
//                                            do self <-- link })
//                       return! waitForUrl(visited.Add(url)) }
//        waitForUrl(Set.empty))

let rec urlCollector =
    MailboxProcessor.SpawnAgent((fun url (visited:Set<string>) ->
                    if visited.Count < limit then
                        if not (visited.Contains(url)) then
                           Async.Start
                                   (async { let! links = collectLinks url
                                            for link in links do
                                                do urlCollector.Post(link) })
                        visited.Add(url)
                    else
                        failwith("Need to continue"))
                , Set.empty, errorHandler =  fun ex msg state -> ContinueProcessing(state))
                    
// ----------------------------

urlCollector.Post("http://news.google.com")

