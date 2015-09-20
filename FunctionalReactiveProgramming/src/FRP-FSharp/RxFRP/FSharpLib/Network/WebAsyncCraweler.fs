namespace Easj360FSharp

open System
open System.Collections.Concurrent
open System.Collections.Generic
open System.IO
open System.Net
open System.Text.RegularExpressions

module WebAsyncCraweler =

    type Message =
        | Done
        | Mailbox of MailboxProcessor<Message>
        | Stop
        | Url of string option

    // Gates the number of crawling agents.
    [<Literal>]
    let Gate = 5

    // Extracts links from HTML.
    let extractLinks html =
        let pattern1 = "(?i)href\\s*=\\s*(\"|\')/?((?!#.*|/\B|mailto:|location\.|javascript:)[^\"\']+)(\"|\')"
        let pattern2 = "(?i)^https?"
 
        let links =
            [
                for x in Regex(pattern1).Matches(html) do
                    yield x.Groups.[2].Value
            ]
            |> List.filter (fun x -> Regex(pattern2).IsMatch(x))
        links
    
    // Fetches a Web page.
    let fetch (url : string) =
        try
            let req = WebRequest.Create(url) :?> HttpWebRequest
            req.UserAgent <- "Mozilla/5.0 (Windows; U; MSIE 9.0; Windows NT 9.0; en-US)"
            req.Timeout <- 5000
            use resp = req.GetResponse()
            let content = resp.ContentType
            let isHtml = Regex("html").IsMatch(content)
            match isHtml with
            | true -> use stream = resp.GetResponseStream()
                      use reader = new StreamReader(stream)
                      let html = reader.ReadToEnd()
                      Some html
            | false -> None
        with
        | _ -> None
    
    let collectLinks url =
        let html = fetch url
        match html with
        | Some x -> extractLinks x
        | None -> []

//    open Helpers

    let crawl url limit =
        // Concurrent queue for saving collected urls.
        let q = ConcurrentQueue<string>()
    
        // Holds crawled URLs.
        let set = HashSet<string>()

        let supervisor =
            MailboxProcessor.Start(fun x ->
                let rec loop run =
                    async {
                        let! msg = x.Receive()
                        match msg with
                        | Mailbox(mailbox) -> 
                            let count = set.Count
                            if count < limit - 1 && run then 
                                let url = q.TryDequeue()
                                match url with
                                | true, str -> if not (set.Contains str) then
                                                    let set'= set.Add str
                                                    mailbox.Post <| Url(Some str)
                                                    return! loop run
                                                else
                                                    mailbox.Post <| Url None
                                                    return! loop run

                                | _ -> mailbox.Post <| Url None
                                       return! loop run
                            else
                                mailbox.Post Stop
                                return! loop run
                        | Stop -> return! loop false
                        | _ -> printfn "Supervisor is done."
                               (x :> IDisposable).Dispose()
                    }
                loop true)

    
        let urlCollector =
            MailboxProcessor.Start(fun y ->
                let rec loop count =
                    async {
                        let! msg = y.TryReceive(6000)
                        match msg with
                        | Some message ->
                            match message with
                            | Url u ->
                                match u with
                                | Some url -> q.Enqueue url
                                              return! loop count
                                | None -> return! loop count
                            | _ ->
                                match count with
                                | Gate -> supervisor.Post Done
                                          (y :> IDisposable).Dispose()
                                          printfn "URL collector is done."
                                | _ -> return! loop (count + 1)
                        | None -> supervisor.Post Stop
                                  return! loop count
                    }
                loop 1)
    
        /// Initializes a crawling agent.
        let crawler id =
            MailboxProcessor.Start(fun inbox ->
                let rec loop() =
                    async {
                        let! msg = inbox.Receive()
                        match msg with
                        | Url x ->
                            match x with
                            | Some url -> 
                                    let links = collectLinks url
                                    printfn "%s crawled by agent %d." url id
                                    for link in links do
                                        urlCollector.Post <| Url (Some link)
                                    supervisor.Post(Mailbox(inbox))
                                    return! loop()
                            | None -> supervisor.Post(Mailbox(inbox))
                                      return! loop()
                        | _ -> urlCollector.Post Done
                               printfn "Agent %d is done." id
                               (inbox :> IDisposable).Dispose()
                        }
                loop())

        // Spawn the crawlers.
        let crawlers = 
            [
                for i in 1 .. Gate do
                    yield crawler i
            ]
    
        // Post the first messages.
        crawlers.Head.Post <| Url (Some url)
        crawlers.Tail |> List.iter (fun ag -> ag.Post <| Url None)

// Example:
//crawl "http://news.google.com" 25

// Output:
// http://news.google.com crawled by agent 1.
// http://www.gstatic.com/news/img/favicon.ico crawled by agent 2.
// http://www.google.com/webhp?hl=en&tab=nw crawled by agent 5.
// http://www.google.com/imghp?hl=en&tab=ni crawled by agent 3.
// http://video.google.com/?hl=en&tab=nv crawled by agent 4.
// http://www.google.com/prdhp?hl=en&tab=nf crawled by agent 5.
// http://news.google.com/news?pz=1&amp;cf=all&amp;ned=us&amp;hl=en
// &amp;topic=h&amp;num=3&amp;output=rss crawled by agent 1.
// http://www.google.com/intl/en/options/ crawled by agent 4.
// http://maps.google.com/maps?hl=en&tab=nl crawled by agent 2.
// http://www.google.com/finance?hl=en&tab=ne crawled by agent 1.
// http://scholar.google.com/schhp?hl=en&tab=ns crawled by agent 2.
// http://www.google.com/realtime?hl=en&tab=nY crawled by agent 2.
// http://mail.google.com/mail/?tab=nm crawled by agent 3.
// http://books.google.com/bkshp?hl=en&tab=np crawled by agent 5.
// http://translate.google.com/?hl=en&tab=nT crawled by agent 4.
// http://blogsearch.google.com/?hl=en&tab=nb crawled by agent 1.
// http://www.google.com/calendar?hl=en&tab=nc crawled by agent 3.
// http://picasaweb.google.com/home?hl=en&tab=nq crawled by agent 5.
// http://www.google.com/reader/?tab=ny crawled by agent 1.
// http://docs.google.com/?tab=no crawled by agent 4.
// https://www.google.com/accounts/ServiceLogin?service=news&amp;pas
// sive=1209600&amp;continue=http://news.google.com/&amp;followup=htt
// p://news.google.com/ crawled by agent 1.
// Agent 1 is done.
// http://www.google.com/preferences?hl=en&prev=http://news.google.com/ crawled by agent 4.
// Agent 4 is done.
// http://sites.google.com/?tab=n3 crawled by agent 3.
// Agent 3 is done.
// http://www.youtube.com/?hl=en&tab=n1 crawled by agent 2.
// Agent 2 is done.
// http://groups.google.com/grphp?hl=en&tab=ng crawled by agent 5.
// Agent 5 is done.
// URL collector is done.
// Supervisor is done.