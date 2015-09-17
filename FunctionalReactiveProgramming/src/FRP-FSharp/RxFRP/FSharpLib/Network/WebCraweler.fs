module WebCraweler 


#if INTERACTIVE
#r "FSharp.PowerPack.dll"
#endif

open System.Net
open Microsoft.FSharp.Control.WebExtensions
open System.Text.RegularExpressions
open System.Threading
open System.IO

let limit = 1000
let linkPattern = "href=\s*\"[^\"h]*(http://[^&\"]*)\""
let getLinks text = 
  [for link in Regex.Matches(text, linkPattern) -> link.Groups.[1].Value]

let (<--) (m:MailboxProcessor<_>) msg = m.Post(msg)

let guiContext = SynchronizationContext.Current


type RequestGate(n) =
  let semaphore = new Semaphore(n, n)
  member this.AsyncAcquire(?timeout) =
    async { let! ok = Async.AwaitWaitHandle(semaphore, ?millisecondsTimeout=timeout)
            if ok then
              return { new System.IDisposable with
                         member this.Dispose() =
                           semaphore.Release() |> ignore }
            else return! failwith "Couldn't acquire semaphore" }
            
let webRequestGate = RequestGate(System.Environment.ProcessorCount)

let mutable urlVisted = 0
let analyzeUrl webAddress = 
        async
            {
                let keeptrace = System.Threading.Interlocked.Increment(&urlVisted)                
                printf "%s\n" webAddress
            }

let rec collectLinks (url:string) =
  async {
          let! html =
            async { use! holder = webRequestGate.AsyncAcquire()
                    let request = WebRequest.Create(url, Timeout=5)
                    use! response = request.AsyncGetResponse()
                    use reader = new StreamReader(response.GetResponseStream())
                    return! reader.AsyncReadToEnd() }
          
          let links = getLinks html
          do! Async.SwitchToContext guiContext
          for l in links do Async.Start( analyzeUrl l ) 
          do! Async.SwitchToThreadPool()
          return links }
          
let urlCollector = 
  MailboxProcessor.Start(fun inbox ->
    let rec waitForUrl (visited:Set<string>) =
      async { if visited.Count < limit then
                let! url = inbox.Receive()
                if not <| Set.contains url visited then
                  do Async.Start ( 
                        async { let! links = collectLinks url
                                for link in links do inbox.Post(link) } ) // <-- link } )
                return! waitForUrl (Set.add url visited) }
                
    waitForUrl Set.empty)           
    
let urlCollectorNoSet = 
  MailboxProcessor.Start(fun inbox ->
    let rec waitForUrl =
      async { 
                let! url = inbox.Receive()
                do Async.Start ( 
                        async { let! links = collectLinks url
                                for link in links do inbox.Post(link) } ) // <-- link } )
                return! waitForUrl  }
                
    waitForUrl )           
    
         

let feeds = 
   [
    "http://blogs.msdn.com/MainFeed.aspx?Type=AllBlogs";
    "http://msmvps.com/blogs/MainFeed.aspx";
    "http://weblogs.asp.net/MainFeed.aspx";
    "http://www.bing.com/news";
    "http://www.libero.it";
    "http://www.cnn.com";
    "http://news.google.com"
   ]
   
feeds |> List.map (fun x -> urlCollector.Post(x) ) |> ignore  // urlCollector <-- x




(*
open System.Collections.Generic 
open System.Net 
open System.IO 
open System.Threading 
open System.Text.RegularExpressions 
 
let limit = 10     
 
let linkPat = "href=\s*\"[^\"h]*(http://[^&\"]*)\"" 
let getLinks (txt:string) = 
    [ for m in Regex.Matches(txt,linkPat)  -> m.Groups.Item(1).Value ] 
 
let (<--) (mp: MailboxProcessor<_>) x = mp.Post(x) 
 
// A type that helps limit the number of active web requests 
type RequestGate(n:int) = 
    let semaphore = new Semaphore(initialCount=n,maximumCount=n) 
    member x.AcquireAsync(?timeout) = 
        async { let! ok = semaphore.AsyncWaitOne(?millisecondsTimeout=timeout) 
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
                        use reader = new StreamReader( 
                            response.GetResponseStream()) 
 
                        // Read the response stream 
                        return! reader.AsyncReadToEnd()  } 
 
            // Compute the links, synchronously 
            let links = getLinks html 
 
            // Report, synchronously 
            do printfn "finished reading %s, got %d links"  
                    url (List.length links) 
 
            // We're done 
            return links } 
 
let urlCollector = 
    MailboxProcessor.Start(fun self -> 
        let rec waitForUrl (visited : Set<string>) = 
            async { if visited.Count < limit then 
                        let! url = self.Receive() 
                        if not (visited.Contains(url)) then 
                            Async.Start  
                                (async { let! links = collectLinks url 
                                         for link in links do 
                                             do self <-- link }) 
                        return! waitForUrl(visited.Add(url)) } 
 
        waitForUrl(Set.Empty)) 
 
urlCollector <-- "http://news.google.com" 
// wait for keypress to end program 
System.Console.ReadKey() |> ignore 
*)