namespace Easj360FSharp 
//#light
//
//open System
//open System.IO
//open System.Net
//open System.Net.Sockets
//open Microsoft.FSharp.Control
//
////open Async2.TypeExtensions
//
//(* gets a web page using System.Net.Sockets.TcpClient *)
//let wgetTcp (url: string) = 
//  async { let uri = Uri(url)
//          let! host = Dns.GetHostEntryAsync(uri.DnsSafeHost)
//          use sock = new TcpClient()
//          do! sockConnectAsync(host.AddressList, uri.Port)
//          use ms = new MemoryStream()
//          use sw = new StreamWriter(ms)
//          do sw.WriteLine("GET {0} HTTP/1.0", uri.PathAndQuery)
//          do sw.WriteLine("Host: {0}", uri.Host)
//          do sw.WriteLine()
//          do sw.Close()
//          use s = sock.GetStream() :> Stream
//          do! s.WriteAllBytes(ms.ToArray())
//          let! b = s.ReadAllBytes()
//          use ms = new MemoryStream(b)
//          use sr = new StreamReader(ms)
//          let t = sr.ReadToEnd() 
//          return t }
//  
//(* gets a web page using System.Net.WebRequest which doesn't work because BeginGetWebResponse blocks *)
//let wgetWebReq (url: string) =
//  async { let w = WebRequest.Create(url)
//          use! r = w.GetWebResponseAsync()
//          use s = r.GetResponseStream()
//          let! b = s.ReadAllBytes()
//          use ms = new MemoryStream(b)
//          use sr = new StreamReader(ms)
//          let t = sr.ReadToEnd() 
//          return t }
//
//open System.Text.RegularExpressions
//
//(* extracts href links from an html document *)
//let getHrefs (page : string) = 
//  let re = Regex(@"a href=""(http://[^""]*)""", RegexOptions.IgnoreCase ||| RegexOptions.Compiled)
//  let ms = re.Matches(page); 
//  [ for m : Match in Seq.untyped_to_typed ms -> m.Groups.get_Item(1).Value ]
//
//open System.Windows.Forms
//
//type UrlItem =
//  val url : string
//  val mutable status : string
//  val mutable error : string
//  new(url) = { url = url; status = "unstarted"; error = null }
//
//type Gui () =
//  let form = new Form()
//  let dg = new DataGrid();
//  do dg.Dock <- DockStyle.Fill
//  do form.Controls.Add(dg)
//  let update _ = dg.Refresh()
//  let timer = new Timer()
//  do timer.Interval <- 1000
//  do timer.Tick.Add update
//  do form.Show()
//  
//  member this.Form = form
//  member this.SetDataSource(ds : seq<UrlItem>) = dg.DataSource <- ds
//  member this.Start() = timer.Start()
//  member this.Stop() = timer.Stop()
//
//let wget = wgetTcp
//
//let crawl items = 
//  let f (item : UrlItem) = 
//    async {
//      do item.status <- "connecting"
//      do Console.WriteLine("connecting {0}", item.url)
//      let a = 
//        async { 
//          try 
//            let! r = wget item.url 
//            do item.status <- "finished"
//            do Console.WriteLine("finished {0}", item.url)
//            return r 
//          with
//            e -> 
//              do item.status <- "error"
//              do item.error <- e.Message
//              do Console.WriteLine("error {0}", item.url)
//              return null }
//      let! r = Async2.timed a 30000.0
//      match r with
//        | None -> do item.status <- "timeout"
//                  do Console.WriteLine("timeout {0}", item.url)
//                  return ()
//        | Some _ -> return () }
//  Async2.parallelIter f (items |> List.of_array)
//
//let webcrawl () =
//  let page = wget "http://news.google.com/" |> Async2.runOne
//  let items = getHrefs page |> List.map (fun url -> UrlItem(url)) |> List.to_array
//  let g = new Gui()
//  do g.SetDataSource(items)
//  g.Start()
//  System.Threading.ThreadPool.QueueUserWorkItem(fun _ -> crawl items |> Async2.runOne; g.Stop(); Console.WriteLine("done")) |> ignore
//  Application.Run(g.Form)
