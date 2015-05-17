
open System
open System.IO
open System.Net
open System.Text.RegularExpressions

let downloadWebpage (url : string) =
    let req = WebRequest.Create(url)
    let resp = req.GetResponse()
    let stream = resp.GetResponseStream()
    let reader = new StreamReader(stream)
    reader.ReadToEnd()

let extractImageLinks html =
    let results = Regex.Matches(html, "<img src=\"([^\"]*)\"")
    [
        for r in results do
            for grpIdx = 1 to r.Groups.Count - 1 do
                yield r.Groups.[grpIdx].Value
    ] |> List.filter (fun url -> url.StartsWith("http://"))

let downloadToDisk (url : string) (filePath : string) =
    use client = new System.Net.WebClient()
    client.DownloadFile (url, filePath)

let scrapeWebsite destPath (imageUrls : string list) =
    imageUrls
    |> List.map(fun url ->
            let parts = url.Split( [| '/' |] )
            url, parts.[parts.Length - 1])
    |> List.iter(fun (url, filename) ->
            printfn "Dowanloading image %s" (Path.GetFileNameWithoutExtension(filename))
            downloadToDisk url (Path.Combine(destPath, filename)))


//This refactoring would enable you to use the web scrapper in the following fashion:
downloadWebpage "http://oreilly.com/"
|> extractImageLinks
|> scrapeWebsite @"C:\Temp"

