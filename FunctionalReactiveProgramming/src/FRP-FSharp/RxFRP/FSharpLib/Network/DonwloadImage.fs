namespace Easj360FSharp

open System.IO
open System.Net
open System.Text.RegularExpressions

module DownloadImage = 

    let url = @"http://oreilly.com/"
    let req = WebRequest.Create(url)
    let resp = req. GetResponse()
    let stream = resp.GetResponseStream()
    let reader = new StreamReader(stream)
    let html = reader.ReadToEnd()

        // Extract all images
    let results = Regex.Matches(html, "<img src=\"([^\"]*)\"")
    let allMatches =
        [
            for r in results do
                for grpIdx = 1 to r.Groups.Count - 1 do
                    yield r.Groups.[grpIdx].Value
        ]
    let fullyQualified =
        allMatches
        |> List. filter (fun url -> url.StartsWith("http://"))
    // Download the images
    let downloadToDisk (url : string) (filePath : string) =
        use client = new System.Net.WebClient()
        client. DownloadFile (url, filePath)
    fullyQualified
    |> List. map(fun url -> 
                    let parts = url.Split( [| '/' |] )
                    url, parts.[parts.Length - 1])
    |> List. iter(fun (url, filename) -> downloadToDisk url (@"c:\Temp\" + filename))
