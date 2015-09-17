// 10-15
// download and calculate length of info on homepage for.. 
open System.Net

let urlList = [ "Microsoft.com", "http://www.microsoft.com/" 
                "MSDN", "http://msdn.microsoft.com/" 
                "Google", "http://www.google.com"
              ]

let fetchAsync(name, url:string) =
    let uri = new System.Uri(url)

    async { 
        let webClient = new WebClient()
        let! html = webClient.AsyncDownloadString(uri)
        printfn "Read %d characters for %s" html.Length name
    }

let runAll() =
    urlList
    |> Seq.map fetchAsync
    |> Async.Parallel 
    |> Async.RunSynchronously
    |> ignore

runAll()
