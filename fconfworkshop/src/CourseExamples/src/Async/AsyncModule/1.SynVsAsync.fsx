open System.IO
open System
open System.Net


// ===========================================
// Async vs Sync
// ===========================================

let httpsync url =
    let req =  WebRequest.Create(Uri url)
    use resp = req.GetResponse()
    use stream = resp.GetResponseStream()
    use reader = new StreamReader(stream)
    let contents = reader.ReadToEnd()
    contents.Length

let httpasync url =
    async { let req =  WebRequest.Create(Uri url)
            use! resp = req.AsyncGetResponse()
            use stream = resp.GetResponseStream()
            use reader = new StreamReader(stream)
            let contents = reader.ReadToEnd()
            return contents.Length }
 
let sites = [
    "http://www.bing.com"; 
    "http://www.google.com"; 
    "http://www.yahoo.com";
    "http://www.facebook.com"; 
    "http://www.microsoft.com" ]
 
#time "on"

let htmlOfSitesSync =
    [for site in sites -> httpsync site]

let htmlOfSites =
    sites
    |> Seq.map (httpasync)
    |> Async.Parallel
    |> Async.RunSynchronously
    


