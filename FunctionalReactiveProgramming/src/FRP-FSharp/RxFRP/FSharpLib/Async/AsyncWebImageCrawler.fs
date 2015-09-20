namespace Easj360FSharp

//open PowerEvent
open System.Net
open Microsoft.FSharp.Control.WebExtensions
open System.Text.RegularExpressions
open System.Threading
open System.Data
open System.Data.SqlClient
open System.Configuration
open System.Xml
open System.IO
open Microsoft.FSharp.Control
open Microsoft.FSharp.Control.WebExtensions
open System.Security.Cryptography
open System

module WebImageModule =


    [<Sealed>]
    type PowerEvent<'del, 'args
         when 'del :  not struct                  
         and 'del :  delegate<'args, unit>  
         and 'del :> Delegate       
         and 'del :  null>() =      
     
        [<DefaultValue>]  
        val mutable private target : 'del   

        static let invoker : Action<_,_,_> =     
            downcast Delegate.CreateDelegate(         
                typeof<Action<'del, obj, 'args>>,         
                typeof<'del>.GetMethod "Invoke")  
        
        member self.Trigger (sender: obj, args: 'args) =     
            match self.target with
            null    -> ()   
            | handler -> invoker.Invoke (handler, sender, args)   
    
        member self.TriggerAsync (sender: obj, args: 'args) =     
            match self.target with     
            null    -> ()   
            | handler ->         async { invoker.Invoke (handler, sender, args) }         
                                |> Async.Start   
                        
        member self.TriggerParallel (sender: obj, args: 'args) =     
            match self.target with     
            null    -> ()   
            | handler ->         handler.GetInvocationList ()      
                                 |> Array.map (fun h -> async {         
                                            invoker.Invoke (downcast h, sender, args) })      
                                 |> Async.Parallel      
                                 |> Async.Ignore      
                                 |> Async.Start   

        interface IDelegateEvent<'del> with      
            member self.AddHandler handler =       
                self.target <- downcast            
                    Delegate.Combine (self.target, handler)      
    
            member self.RemoveHandler handler =       
                self.target <- downcast            
                   Delegate.Remove (self.target, handler)   

        member self.Publish = self :> IDelegateEvent<'del>   
    
        member self.PublishSync =   
            { new IDelegateEvent<'del> with      
                member __.AddHandler handler =       
                    lock self (fun() ->            
                                self.target <- downcast                 
                                    Delegate.Combine (self.target, handler))      
    
                member __.RemoveHandler handler =       
                    lock self (fun() ->            
                        self.target <- downcast                 
                            Delegate.Remove (self.target, handler)) }   
                

        member self.PublishLockFree =   
            { new IDelegateEvent<'del> with      
                    member __.AddHandler handler =       
                        let rec loop o =         
                            let c = downcast Delegate.Combine (o, handler)         
                            let r = Interlocked.CompareExchange(&self.target,c,o)         
                            if obj.ReferenceEquals (r, o) = false then loop r       
                        loop self.target      
                    
                    member __.RemoveHandler handler =       
                        let rec loop o =         
                            let c = downcast Delegate.Remove (o, handler)         
                            let r = Interlocked.CompareExchange(&self.target,c,o)         
                            if obj.ReferenceEquals (r, o) = false then loop r       
                        loop self.target }

    type internal System.Data.SqlClient.SqlCommand with   
            member x.ExecuteNonQueryAsync() =
              Async.FromBeginEnd(x.BeginExecuteNonQuery, x.EndExecuteNonQuery)

    type ImageDonwloadedEvent(url:string) =
            inherit System.EventArgs()
                member this.Url = url

    type ImageDonwloadedHandler = delegate of obj * ImageDonwloadedEvent -> unit
        
    type JobData = int * string * AsyncReplyChannel<string[]>

    type Message =  Start
                    |   Stop
                    |   Pause of int
                    |   QueueJob of string
                    |   Job of JobData 
    
    type WebScrapper(workers:int) =
        let linkPattern = "href=\s*\"[^\"h]*(http://[^&\"]*)\""
        let imgPattern = "<img src=\"([^\"]*)\""
        let isValidUrl = "((https?):((//)|(\\\\))+[\w\d:#@%/;$()~_?\+-=\\\.&]*)"

        let powerImagesDownloaded = PowerEvent<ImageDonwloadedHandler, ImageDonwloadedEvent>()
               
        let getComponent text pattern = 
                async {
                        let components = 
                            [for c in Regex.Matches(text, pattern, RegexOptions.Compiled) -> c.Groups.[1].Value]
                            |> List.filter (fun x -> Regex.IsMatch(x, isValidUrl))
                        let result = components |> List.toArray
                        return result   }

        let getImgs text = getComponent text imgPattern    
        let getLinks text = getComponent text linkPattern  
        
        
        let connectionString = @"Integrated Security=SSPI;Persist Security Info=False;Initial Catalog=WebImageDb;Data Source=I7-PC\Sql2008;Asynchronous Processing=True;"
        
        let insertImageData (name:string, sourceUrl:string, hash:byte[], ext:string, size:int64, content:byte[]) = async {
            try
                use conn = new SqlConnection (connectionString)
                conn.Open()
                use cmd = new SqlCommand("InsertNewImage", conn)
                cmd.CommandType <- CommandType.StoredProcedure
                ignore ( cmd.Parameters.AddWithValue("@Name", name) )
                ignore ( cmd.Parameters.AddWithValue("@SourceUrl", sourceUrl) )        
                ignore ( cmd.Parameters.AddWithValue("@Hash", hash) )
                ignore ( cmd.Parameters.AddWithValue("@Ext", ext) )
                ignore ( cmd.Parameters.AddWithValue("@Size", size) )
                ignore ( cmd.Parameters.AddWithValue("@Content", content) )    
                let sqlResult = new SqlParameter();
                sqlResult.ParameterName <- "@Result"
                sqlResult.SqlDbType <- SqlDbType.Int    
                sqlResult.Direction <- ParameterDirection.Output
                ignore ( cmd.Parameters.Add(sqlResult) )
                let! resCmd = cmd.ExecuteNonQueryAsync()
                let result = (cmd.Parameters.["@Result"].Value :?> int)
                match result with
                | 0 -> return false
                | _ -> return true
            with
            | ex -> let msg = ex.Message
                    return false
        }
          
        let copyImageToMemStream (stream:Stream)(buff:byte[]) =
            async {
                    use ms = new MemoryStream()
                    let rec copy (buff:byte[]) =
                        async{
                                let! read = stream.AsyncRead(buff,0,buff.Length)
                                if read = 0 then 
                                    return ms.ToArray()
                                else 
                                    do! ms.AsyncWrite(buff,0,read)
                                    return! copy(buff)
                        }
                    return! copy(buff)  }

        let downloadImageData (url:string, buff:byte[]) =
            async {    
                        try
                            let  client = System.Net.WebRequest.Create(new System.Uri(url))
                            client.Timeout <- 15000 // 1000 * sec
                            use! resp = client.AsyncGetResponse()
                            use stream = resp.GetResponseStream()
                            let! buffer = copyImageToMemStream(stream)(buff)                                                       
                            return Some(buffer)
                        with
                            _ -> return None    }

        let hashImageData (data:byte[]) =
            async {       
                    use hashFunction = new SHA512Managed()         
                    return hashFunction.ComputeHash(data)   }

        let readHtmlContent (url:string) = 
            async {          
                try
                    let request = WebRequest.Create(new System.Uri(url))
                    use! response = request.AsyncGetResponse()
                    use reader = new StreamReader(response.GetResponseStream())
                    let! readContent =reader.AsyncReadToEnd()
                    return Some(readContent)
                with 
                    _ -> return None    }

        let getExtension name = try
                                     System.IO.Path.GetExtension(name)
                                with
                                | _ -> name

        let storeImage (url:string, buff:byte[]) =
            async {
                    let! data = downloadImageData (url, buff)
                    match data with
                    | Some(data)    ->  try
                                            let! hashedData = hashImageData data
                                            let parts = url.Split('/', '\\')      
                                            let name = parts.[parts.Length - 1]
                                            let ext = getExtension name
                                            let sourceUrl = System.String.Join("/", parts.[0.. parts.Length - 2])
                                            //let! insertImageData(name, sourceUrl, hashedData, ext, data.LongLength, data)
                                            let isImageInserted = true
                                            return isImageInserted
                                        with
                                        | ex -> return false
                    | _             ->  return false    }
       
        let agentManager n f = 
            MailboxProcessor<Message>.Start(fun inbox ->
                //let workers = Array.init n (fun i -> (MailboxProcessor<AgentMessage>.Start(f), new CancellationTokenSource()))
                let jobWorkers = Array.init n (fun i -> MailboxProcessor<Message>.Start(f))
                let rec loop (state:Message) i = async {
                    let! msg = inbox.Receive()                    
                    match msg with
                    | Pause wait->  let! sleep = Async.Sleep(wait)
                                    [0..workers-1] |> List.iter (fun x -> do jobWorkers.[x].Post(Pause(wait)))
                                    return! loop (Pause wait) i                   
                    | QueueJob url   -> let links = jobWorkers.[i].PostAndAsyncReply(fun replayChannel -> Job(n, url, replayChannel))
                                        //links |> Array.iter (fun link -> inbox.Post(QueueJob(link)))
                                        Async.StartWithContinuations(links, 
                                                    (fun l ->  l |> Array.iter (fun link -> inbox.Post(QueueJob(link)))),
                                                    (fun error -> ()),
                                                    (fun cancel -> ()))   
                    | Stop        ->  [0..workers-1] |> List.iter (fun x -> do jobWorkers.[x].Post(Stop))
                                      return () 
                    | message     ->  [0..workers-1] |> List.iter (fun x -> do jobWorkers.[x].Post(message))
                    return! loop state ((i + 1) % n)   }
                loop Start 0  )

        let agent =
            agentManager workers (fun inbox ->
                let rec loop state (jobs : System.Collections.Generic.Queue<JobData>, buff:byte[]) = async {
                    if state = Start then
                        while jobs.Count > 0 do
                            let (id:int, url:string, replyChannel:AsyncReplyChannel<string[]>)  = jobs.Dequeue()
                            let! html = readHtmlContent url
                            if html.IsSome then
                                let! imgs = getImgs html.Value
                                for img in imgs do
                                    let! storageResult = storeImage(img, buff)
                                    if storageResult = true then
                                            let args = ImageDonwloadedEvent(img)
                                            powerImagesDownloaded.TriggerAsync(System.Threading.Thread.CurrentThread.ManagedThreadId, args)
                                let! links = getLinks html.Value
                                replyChannel.Reply(links)                                       
                    let! msg = inbox.Receive()
                    match msg with
                    | Start         ->  return! loop Start (jobs, buff)  
                    | Pause wait    ->  let! sleep = Async.Sleep(wait)
                                        return! loop (Pause wait) (jobs, buff)
                    | Stop          ->  return ()                    
                    | Job jData     ->  jobs.Enqueue(jData)
                                        return! loop state (jobs, buff)
                    | message       ->  return! loop message (jobs, buff)                      
                    }
                loop Start ( new System.Collections.Generic.Queue<JobData>(), Array.zeroCreate<byte> 1024)     )
                 
        [<CLIEvent>]
        member this.EventImagesDownloadedLock = powerImagesDownloaded.PublishLockFree

        member x.Start (feed:seq<string>) =              
            Async.Start( async { Seq.toList feed |>              
                                 List.map (fun f -> agent.Post(QueueJob(f)) ) |> ignore }  )
        
        member x.Resume() = agent.Post(Start)    
        member x.Stop() = agent.Post(Stop)    
        member x.Pause(wait) = agent.Post(Pause(wait))    
        member x.QueueJob(f) = agent.Post(QueueJob f)
                
       

