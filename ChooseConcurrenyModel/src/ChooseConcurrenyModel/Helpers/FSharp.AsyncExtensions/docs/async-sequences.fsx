namespace FSharp
#r @"C:\Tomas\Projects\FSharp\FSharp.AsyncExtensions\bin\FSharp.AsyncExtensions.dll"
#r @"C:\Tomas\Projects\FSharp\FSharp.AsyncExtensions\bin\HtmlAgilityPack.dll"

open FSharp.Control
open System.Net
open System

module Control =
  // [snippet:Definition]
  /// Asynchronous computation that produces either end of a sequence
  /// (Nil) or the next value together with the rest of the sequence.
  type AsyncSeq<'T> = Async<AsyncSeqInner<'T>> 
  and AsyncSeqInner<'T> =
    | Nil
    | Cons of 'T * AsyncSeq<'T>
  // [/snippet]

  type LazySeq<'T> = Lazy<LazySeqInner<'T>> 
  and LazySeqInner<'T> =
    | Nil
    | Cons of 'T * LazySeq<'T>

  let rec nums n : LazySeq<int> =
    lazy LazySeqInner.Cons(n, nums (n + 1))

  let rec map f (ls : LazySeq<_>) = 
    lazy match ls.Value with 
         | Nil -> Nil
         | Cons(h, t) -> LazySeqInner.Cons(f h, map f t)

module Samples =  

  // [snippet:computation expressions #1]
  // When accessed, generates numbers 1 and 2. The number 
  // is returned 1 second after value is requested.
  let oneTwo = asyncSeq { 
    do! Async.Sleep(1000)
    yield 1
    do! Async.Sleep(1000)
    yield 2 }
  // [/snippet]

  // [snippet:computation expressions #2]
  let urls = 
    [ "http://bing.com"; "http://yahoo.com"; 
      "http://google.com"; "http://msn.com" ]

  // Asynchronous sequence that returns URLs and lengths
  // of the downloaded HTML. Web pages from a given list
  // are downloaded synchronously in sequence.
  let pages = asyncSeq {
    use wc = new WebClient()
    for url in urls do
      try
        let! html = wc.AsyncDownloadString(Uri(url))
        yield url, html.Length 
      with _ -> 
        yield url, -1 }    
  // [/snippet]

  // [snippet:using from async]
  // Asynchronous workflow that prints results
  async {
    for url, length in pages do
      printfn "%s (%d)" url length }
  |> Async.Start
  // [/snippet]

  // [snippet:combinators]
  // Print URL of pages that are smaller than 50k
  let printPages =
    pages 
    |> AsyncSeq.filter (fun (_, len) -> len < 50000)
    |> AsyncSeq.map fst
    |> AsyncSeq.iter (printfn "%s")
  
  printPages |> Async.Start
  // [/snippet]

  // [snippet:combinators internals]
  /// Return elements for which the predicate returns true
  let filter f (input : AsyncSeq<'T>) = asyncSeq {
    for v in input do
      if f v then yield v }

  /// Return elements for which the asynchronous predicate returns true
  let filterAsync f (input : AsyncSeq<'T>) = asyncSeq {
    for v in input do
      let! b = f v
      if b then yield v }
  // [/snippet]

module Crawler =

  open System.Text.RegularExpressions
  open HtmlAgilityPack

  /// Asynchronously download the document and parse the HTML
  let downloadDocument url = async {
    try let wc = new WebClient()
        let! html = wc.AsyncDownloadString(Uri(url))
        let doc = new HtmlDocument()
        doc.LoadHtml(html)
        return Some doc 
    with _ -> return None }

  /// Extract all links from the document that start with "http://"
  let extractLinks (doc:HtmlDocument) = 
    try
      [ for a in doc.DocumentNode.SelectNodes("//a") do
          if a.Attributes.Contains("href") then
            let href = a.Attributes.["href"].Value
            if href.StartsWith("http://") then 
              let endl = href.IndexOf('?')
              yield if endl > 0 then href.Substring(0, endl) else href ]
    with _ -> []

  /// Extract the <title> of the web page
  let getTitle (doc:HtmlDocument) =
    let title = doc.DocumentNode.SelectSingleNode("//title")
    if title <> null then title.InnerText.Trim() else "Untitled"

  // ----------------------------------------------------------------------------

  // [snippet:crawler #1]
  /// Crawl the internet starting from the specified page.
  /// From each page follow the first not-yet-visited page.
  let rec randomCrawl url = 
    let visited = new System.Collections.Generic.HashSet<_>()

    // Visits page and then recursively visits all referenced pages
    let rec loop url = asyncSeq {
      if visited.Add(url) then
        let! doc = downloadDocument url
        match doc with 
        | Some doc ->
            // Yield url and title as the next element
            yield url, getTitle doc
            // For every link, yield all referenced pages too
            for link in extractLinks doc do
              yield! loop link 
        | _ -> () }
    loop url
  // [/snippet]

  // [snippet:crawler #2]
  // Use AsyncSeq combinators to print the titles of the first 10
  // web sites that are from other domains than bing.com
  randomCrawl "http://news.bing.com"
  |> AsyncSeq.filter (fun (url, title) -> url.Contains("bing.com") |> not)
  |> AsyncSeq.map snd
  |> AsyncSeq.take 10
  |> AsyncSeq.iter (printfn "%s")
  |> Async.Start
  // [/snippet]