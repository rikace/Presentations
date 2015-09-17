namespace Easj360FSharp 

#if INTERACTIVE
#r "FSharp.PowerPack.dll"
#I "C:\Git\Easj360\FSharpLib"
#I "C:\Git\Easj360\FSharpLib\Async"
#I "C:\Git\Easj360\FSharpLib\Agent"
#load "ThrottlingAgent.fs"
#load "ComparableUrl.fs"
open Easj360FSharp
open Easj360FSharp.ThrottlingAgent 

#endif

module WebCraweler2 =

    open System.Net
    open Microsoft.FSharp.Control.WebExtensions
    open System.Text.RegularExpressions
    open System.Threading
    open System.IO
    open Microsoft.FSharp.Control
    open Microsoft.FSharp.Control.WebExtensions
    open ThrottlingAgent
    open ComparableUrl
    open AsyncGate

    let feedsII = 
               [
                "http://blogs.msdn.com/MainFeed.aspx?Type=AllBlogs";
                "http://msmvps.com/blogs/MainFeed.aspx";
                "http://weblogs.asp.net/MainFeed.aspx";
                "http://www.bing.com/news";
                "http://www.libero.it";
                "http://www.cnn.com";
                "http://news.google.com"
               ]

    let uniq a = a |> Set.ofList |> List.ofSeq

    let parallelAgents = ThrottlingAgent(4)
    let gates = AsyncGate.RequestGate(4)

//    let linkPattern = "href=\s*\"[^\"h]*(http://[^&\"]*)\""
//    let imgPattern = "<img src=\"([^\"]*)\""
    //<a rel="Mp4Med" href="http://media.ch9.ms/ch9/5436/8fa29cfe-0a64-4e77-ab9c-739838f55436/Part9DataBindingtotheRecipeDataSourcecs_mid.mp4"></a>    //<a href="http://media.ch9.ms/ch9/5436/8fa29cfe-0a64-4e77-ab9c-739838f55436/Part9DataBindingtotheRecipeDataSourcecs.mp3">MP3</a> <span class="usage">(Audio only)</span>

    let limit = 50
    let linkPattern = "href=\"([^\"]+)"
    let imgPattern = "<img src=\"([^\"]*)\""
    let videoPattern = "<a src=\"([^\"]*).mp4\""

    let (|ParseRegex|_|) regex str =
       let opts = System.Text.RegularExpressions.RegexOptions.Compiled ||| System.Text.RegularExpressions.RegexOptions.IgnoreCase
       let m = System.Text.RegularExpressions.Regex(regex, opts).Match(str)
       if m.Success then 
            Some (List.tail [ for x in m.Groups -> x.Value ])
       else None
     
//    let parseHtml text pattern = async {
//        match text with
//        |  ParseRegex pattern x -> return x
//        | _ -> return [] }

    let parseHtml text pattern = async {
        let links = [for link in Regex.Matches(text, pattern, RegexOptions.Compiled) -> link.Groups.[1].Value ]
        return (uniq links) }

    let extract baseUri html pattern = async {
        let regex = System.Text.RegularExpressions.Regex(pattern, RegexOptions.Compiled)
        let links =  [ for url in regex.Matches html ->
                            try [ComparableUri(System.Uri(baseUri, url.Groups.[1].Value))] with _ -> [] ]
                     |> List.concat |> uniq
        return links }

    let getLinks baseUri text = extract baseUri text linkPattern
    let getImgs baseUri text = extract baseUri text imgPattern
    let getVideos baseUri text = extract baseUri text videoPattern
                    
    let destinationImages = @"U:\Temp"
    //let destinationVideos = ""

    let downloadToDisk (destination:string) (url:ComparableUri) =
        async {    
                    try
                        use  client = new System.Net.WebClient()                
                        let parts = url.AbsoluteUri.Split( [| '/' |] )      
                        let filePath = Path.Combine(destination, parts.[parts.Length - 1])
                        printfn "File %s" filePath
                        if not (System.IO.File.Exists(filePath)) then
                            printfn "Downloading %s ..." url.AbsoluteUri      
                            let r = Async.AwaitEvent client.DownloadFileCompleted
                            client.DownloadFileAsync(url, filePath)                            
                            ignore(r)
                    with
                    | _ -> () }
    
    let downloadToDiskImage = downloadToDisk(destinationImages)
    //let downloadToDiskVideo = downloadToDisk(destinationVideos)

    let parseAndAct html (url:System.Uri) (f:System.Uri -> string -> Async<ComparableUri list>) (p:ComparableUri -> Async<unit>) = async {
                let! imgs = f url html    
                imgs
                |> Seq.map(fun i -> p i)
                |> Seq.iter(fun work -> parallelAgents.DoWork(work)) }

    let collectLinks (url:ComparableUri) = async {
            let! html = async { try
                                    use! gate = gates.Acquire()
                                    let request = WebRequest.Create(url, Timeout=5)
                                    use! response = request.AsyncGetResponse()
                                    use reader = new StreamReader(response.GetResponseStream())
                                    let! text = reader.AsyncReadToEnd()
                                    return Some(text) 
                                with 
                                | ex -> printfn "Error: %s" ex.Message 
                                        return None }
            match html with
            | Some(x) -> do! parseAndAct x url getImgs downloadToDiskImage
                         //do! parseAndAct x url getVideos downloadToDiskVideo 
                         return! getLinks url x
            | _ -> return [] }
                  
    let urlCollector = MailboxProcessor.Start(fun inbox ->
        let runPredicate v = if limit > 0 then (v < limit)
                             else (true)
        let rec waitForUrl(visited: Set<ComparableUri>) = async { 
                     if runPredicate(visited.Count) then 
                        let! url = inbox.Receive()
                        if not(visited.Contains url) then 
                            let! links = collectLinks url   
                            links |> List.iter inbox.Post 
                            return! waitForUrl(visited.Add url) 
                        else return! waitForUrl(visited) }
        waitForUrl(Set.empty) )
 
//urlCollector.Post(ComparableUri(System.Uri "http://www.tube8.com"))

            
///////////////////////////////////////////////////////////

    type RequestGate(n) =
              let semaphore = new Semaphore(n, n)
              member this.AsyncAcquire(?timeout) =
                async { let! ok = Async.AwaitWaitHandle(semaphore, ?millisecondsTimeout=timeout)
                        if ok then
                          return { new System.IDisposable with
                                     member this.Dispose() =
                                       semaphore.Release() |> ignore }
                        else return! failwith "Couldn't acquire semaphore" }
            
            
    type WebCraweler_MailBoxProcessor(destination:string) =

            let linkPattern = "href=\s*\"[^\"h]*(http://[^&\"]*)\""
            let imgPattern = "<img src=\"([^\"]*)\""

      
            let getLinks text = 
                    async {
                            let htmls = [for link in Regex.Matches(text, linkPattern, RegexOptions.Compiled) -> link.Groups.[1].Value]
                            let result = htmls |> uniq
                            return result
                          }

            let getImgs text =
                    async {
                           let imgs = [ for img in Regex.Matches(text, imgPattern, RegexOptions.Compiled) -> img.Groups.[1].Value ]
                           let result =  imgs |> uniq
                           return result
                          }                    

            let webRequestGate = RequestGate(System.Environment.ProcessorCount)

            let createDir (dir:string) = 
                System.IO.Directory.CreateDirectory dir

            let downloadToDisk (url : string) =
                async {    
                            try
                            use  client = new System.Net.WebClient()                
                            let parts = url.Split( [| '/' |] )      
                            let filePath = destination + parts.[parts.Length - 1]
                            if not (System.IO.Directory.Exists(System.IO.Path.GetDirectoryName(filePath))) then 
                                createDir(System.IO.Path.GetDirectoryName(filePath)) |> ignore
                            if not (System.IO.File.Exists(filePath)) then
                                printfn "Downloading %s ..." url         
                                let wait = Async.AwaitEvent client.DownloadDataCompleted          
                                client.DownloadFileAsync (new System.Uri(url), filePath)
                                wait |> ignore
                            with
                                _ -> ignore
                    }

            let parallelWorker n f =
                MailboxProcessor.Start(fun inbox ->
                    let workers = 
                        Array.init n (fun i -> MailboxProcessor.Start(f))
                    let rec loop i = async {
                        let! msg = inbox.Receive()
                        workers.[i].Post(msg)
                        return! loop((i + 1) % n)
                    }
                    loop 0
                )
            
            let agent =
                parallelWorker System.Environment.ProcessorCount (fun inbox ->
                    let rec loop() = async {
                        let! msg = inbox.Receive()
                        let! task = downloadToDisk(msg)
                        return! loop()
                        }
                    loop()
                )
            
            let collectLinks (url:string) =
                async {
                    let! html =
                            async { 
                                        try
                                        use! holder = webRequestGate.AsyncAcquire()
                                        let request = WebRequest.Create(url, Timeout=5)
                                        use! response = request.AsyncGetResponse()
                                        use reader = new StreamReader(response.GetResponseStream())
                                        return! reader.AsyncReadToEnd() 
                                        with 
                                            _ -> return System.String.Empty                     
                                        }
                    if not (System.String.IsNullOrEmpty(html)) then 
                        let! imgs = getImgs html
                        for img in imgs do
                            agent.Post(img)            
                
                    let! links = getLinks html              
                    return links
                    }        
                  
            let urlCollector = 
                MailboxProcessor.Start(fun inbox ->
                let rec waitForUrl  =
                  async { 
                            let! url = inbox.Receive()
                            do Async.Start ( 
                                    async { let! links = collectLinks url                                
                                            for link in links do 
                                               //printfn "link %s ..." link
                                                inbox.Post(link)                                
                                                 } )  // <-- link } )
                            return! waitForUrl }
                waitForUrl )

            member x.Start (feed:seq<string>) =  
                Seq.toList feed |>              
                List.map (fun x -> urlCollector.Post(x) ) |> ignore  
                System.Console.ReadLine()



    ///////////////////////////////////////////////////

    ////#r "FSharp.PowerPack.dll"
    //open System.Net
    //open Microsoft.FSharp.Control.WebExtensions
    //open System.Text.RegularExpressions
    //open System.Threading
    //open System.IO
    //open Microsoft.FSharp.Control
    //open Microsoft.FSharp.Control.WebExtensions

    //type RequestGate(n) =
    //          let semaphore = new Semaphore(n, n)
    //          member this.AsyncAcquire(?timeout) =
    //            async { let! ok = Async.AwaitWaitHandle(semaphore, ?millisecondsTimeout=timeout)
    //                    if ok then
    //                      return { new System.IDisposable with
    //                                 member this.Dispose() =
    //                                   semaphore.Release() |> ignore }
    //                    else return! failwith "Couldn't acquire semaphore" }
    type WebCrawelerChannel9() =
        let linkPattern = "<a href=\"[^\"h]*(/(Blogs|Forums|Shows)[^&\"]*)\""
        let videoPattern = "<a rel=\"Mp4\" href=\s*\"[^\"h]*(http://[^&\"]*)\""
        let titlePattern = @"<title\b[^>]*>(.*?)</title>"

        //let uniq a = a |> Set.ofList |> List.ofSeq

        let getTitle text = async {
            let htmls = [for link in Regex.Matches(text, titlePattern, RegexOptions.Compiled) -> link.Groups.[1].Value]
            let result = htmls |> List.head
            return result }
  
  
        let log msg = printfn "%s" msg
    
        let getValueFromList (l: 'T list) =
                        match l with
                        | n when n.Length > 0 -> Some(n |> List.head)
                        | _ -> None
        let getData text =
                async {
                        let videos = getValueFromList [ for img in Regex.Matches(text, videoPattern, RegexOptions.Compiled) -> img.Groups.[1].Value ]
                        let title = getValueFromList [for link in Regex.Matches(text, titlePattern, RegexOptions.Compiled) -> link.Groups.[1].Value]
                        let htmls = [for link in Regex.Matches(text, linkPattern, RegexOptions.Compiled) -> link.Groups.[1].Value] |> uniq |> List.map(fun f -> "http://channel9.msdn.com/" + f)
                        let result = if htmls.Length > 0 then Some(htmls) else None
                        return (result, videos, title)
                        }                    

        let asyncReadAllBytesFromStreamWithProgress(stream:Stream, length:int, progress:int->unit) = 
                    async {
                        let offset = ref 0
                        let count = ref length
                        let buffer = Array.zeroCreate<byte> !count
                        while !count > 0 do
                            let! bytesRead = stream.AsyncRead(buffer, !offset, (min 524288 !count))
                            if bytesRead = 0 then raise (new EndOfStreamException("Read beyond the EOF"))
                            do offset := !offset + bytesRead
                            do count := !count - bytesRead
                            progress(100 * !offset / length)
                        return buffer
                    }
            
                    
        let webRequestGate = RequestGate(1)

        let getFileName title =
                        let fn = System.Net.WebUtility.HtmlDecode(title)
                        if fn.IndexOf('|') = -1 then fn.Trim().Replace(' ','_')
                        else fn.Substring(0, fn.IndexOf("|")).Trim().Replace(' ','_')

        let downloadToDisk =
            let files = new ResizeArray<string>()
            MailboxProcessor.Start(fun inbox ->
                let rec loop n = async {    
                        try
                            let! (v, t) = inbox.Receive()
                            log("receive Message " + t)
                            if not(files.Contains(t)) then                                        
                                //log("analyzing title >>> " + t)
                                files.Add(t)
                                use  client = new System.Net.WebClient()                
                                let fileName = getFileName t              
                                printfn "Get file info >>> %s" fileName
                                let filePath =  @"H:\Temp\" + fileName + ".mp4"     
                                if not (System.IO.File.Exists(filePath)) then
                                    printfn "Downloading %s ..." v                                  
                                    let r = Async.AwaitEvent client.DownloadFileCompleted
                                    client.DownloadFileAsync(new System.Uri(v), filePath)  
                                    let! comp = r
                                    ()
                            else //log("already analyzed title >>> " + t)
                            return! loop(n + 1)
                        with
                        | ex -> printfn "Error %s" ex.Message
                                return! loop(n)
                    }
                loop(0))
            

        let collectLinks (url:string) =
            async {
                let! html =
                        async {                        
                                try 
                                    //log("Acquiring lock")
                                    use! holder = webRequestGate.AsyncAcquire()                                   
                                    let request = WebRequest.Create(url)
                                    use! response = request.AsyncGetResponse()
                                    use reader = new StreamReader(response.GetResponseStream())
                                    log("Get data for url >>> " + url)
                                    let! txt = reader.AsyncReadToEnd()
                                    return Some(txt) 
                                with 
                                | ex ->   printfn "Error Collect Links >>> url %s >>> %s" url ex.Message
                                          return None  }

                if html.IsNone then 
                     //log("Html empty >>> " + url)
                     return None
                else
                    //log("Getting data >>> " + url)
                    let! (l, v, t) = getData html.Value 
                    //log("Get Data!! >>> " + url)            
                    match (l, v, t) with
                    | l, Some(v), Some(t) -> downloadToDisk.Post(v,t)                                     
                                             return l
                    | l, _, _ -> return l
                }        
                  
        let urlCollector = 
            MailboxProcessor.Start(fun inbox ->
            let rec waitForUrl n  =
                async { 
                        let! url = inbox.Receive()
                        //log("Receive new link >>> " + url)
                        do Async.Start ( 
                                async { let! links = collectLinks url
                                        match links with
                                        | Some(l) ->    log(url + ">>> found match list " + (List.length l).ToString())
                                                        l |> List.iter(fun f -> inbox.Post(f))                               
                                        | None ->       //log(url + ">>> NOT FOUBD MATCH")
                                                        ()
                                                } )   
                        return! waitForUrl (n+1)}
            waitForUrl 0 )
               
        member s.Start (feed:seq<string>) =  
            Seq.toList feed |>              
            List.map (fun x -> urlCollector.Post(x) ) |> ignore  

    //    let urls = [
    //        "http://channel9.msdn.com/Tags/functional+programming";
    //        "http://channel9.msdn.com/Tags/functional+programming?page=2";
    //        "http://channel9.msdn.com/Tags/functional+programming?page=3";
    //        "http://channel9.msdn.com/Tags/functional+programming?page=4";
    //        ]
    //
    //    Start urls

     