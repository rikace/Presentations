namespace Easj360FSharp

module ChachingWebPage =
        open System
        open System.Threading
        open System.Net
        open System.IO
        open System.Xml 
        open Microsoft.FSharp.Control
        open Microsoft.FSharp.Control.WebExtensions

        let requestgate = AsyncGate.RequestGate(Environment.ProcessorCount)
 
        type FeedItem(title:string, description:string) =
            member x.Title = title
            member x.Description = description    
    
        type FeedSearch(feeds:seq<string>) =
            let downloadUrl(url:string) = async {    
                use! gate = requestgate.Acquire()
                let req = HttpWebRequest.Create(url)
                let! resp = req.AsyncGetResponse()
                use rst = resp.GetResponseStream()
                use reader = new System.IO.StreamReader(rst)
                let! str = reader.AsyncReadToEnd()
                return str }
    
            let searchItems(feed:string, keywordArray) = async {
                let! xml = downloadUrl(feed)
                let doc = new XmlDocument()
                doc.LoadXml(xml)
    
                let items = 
                  [ for nd in doc.SelectNodes("rss/channel/item") do
                      let title = nd.SelectSingleNode("title").InnerText 
                      let descr = nd.SelectSingleNode("description").InnerText
                      yield new FeedItem(title, descr) ]
        
                let result =
                  items
                    |> List.filter (fun item ->
                        keywordArray |> Array.exists (fun key ->
                          item.Title.IndexOf(key, StringComparison.OrdinalIgnoreCase) > 0
                          || item.Description.IndexOf(key, StringComparison.OrdinalIgnoreCase) > 0))
                return result }
        
            member x.Search(keyword:string) =
              let keywordsArray = keyword.Split(' ')
              Async.RunSynchronously( 
                    Async.Parallel([ for feed in feeds do
                                     yield searchItems(feed, keywordsArray) ]))
              |> List.concat
              |> Seq.ofList
      
      
        let pages = ["http://moma.org/"; "http://www.thebritishmuseum.ac.uk/"; ]
        let search = FeedSearch pages
        let seqResult = search.Search "test"

        /////////////////////////////////////////////////////


        let AsyncFetch (url:string) = async {
           let r = WebRequest.Create(url)
           let! resp = r.AsyncGetResponse()
           use reader = new StreamReader(resp.GetResponseStream())
           let! html = reader.AsyncReadToEnd()
           do printfn "Read %d chars from %s" html.Length url }
        let work() = pages |> List.iter (AsyncFetch >> Async.Start)

        let httpAsync (url:string) (cont: string -> unit) =
          let req = WebRequest.Create(url) in
          let iar = 
            req.BeginGetResponse((fun iar -> 
              let rsp = req.EndGetResponse(iar) in
              let str = new StreamReader(rsp.GetResponseStream()) in
              let html = str.ReadToEnd() in
              rsp.Close();
              cont html), 0) in
          ()

        //do httpAsync  "http://www.microsoft.com" (fun html -> html.ToString())
        //do httpAsync  "http://www.google.com" (fun html -> show html)
        //
        //let collectUrlsAsync url cont = httpAsync url (getUrls >> cont)
        //do collectUrlsAsync "http://www.microsoft.com" (fun urls -> show urls)
