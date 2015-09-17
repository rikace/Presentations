namespace RssFeedsLibrary

open System
open System.Net
open System.IO
open System.Xml
open Microsoft.FSharp.Control

type FeedItem(title:string, description:string) = 
  member x.Title = title
  member x.Description = description
    
type FeedSearch(feeds:seq<string>) = 
  let downloadUrl(url:string) = async {
    let req = HttpWebRequest.Create(url)
    let! rsp = req.AsyncGetResponse()
    use rst = rsp.GetResponseStream()
    use reader = new StreamReader(rst)
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
            keywordArray |> Array.exists (fun keyword ->
              item.Title.IndexOf(keyword, StringComparison.OrdinalIgnoreCase) > 0
              || item.Description.IndexOf(keyword, StringComparison.OrdinalIgnoreCase) > 0)) 
    return result }

  member x.Search(keywords:string) =
    let keywordsArray = keywords.Split(' ')
    Async.RunSynchronously
      (Async.Parallel([ for feed in feeds do
                          yield searchItems(feed, keywordsArray) ])) 
    |> List.concat
    |> Seq.ofList
    
    
    (*                "http://services.community.microsoft.com/feeds/feed/CSharpHeadlines"
                //"http://blogs.msdn.com/MainFeed.aspx?Type=AllBlogs",
                //"http://msmvps.com/blogs/MainFeed.aspx",
                //"http://weblogs.asp.net/MainFeed.aspx"*)