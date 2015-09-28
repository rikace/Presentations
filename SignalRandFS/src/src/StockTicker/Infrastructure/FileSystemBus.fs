namespace StockTicker

module FileSystemBusModule =

    open System
    open System.Collections.Generic
    open System.IO
    open Microsoft.AspNet.SignalR
    open Microsoft.AspNet.SignalR.Messaging

    type FileSystemBus(resolver:IDependencyResolver, configuration:ScaleoutConfiguration) as this=
        inherit ScaleoutMessageBus(resolver, configuration)
        let basePath = Path.Combine(Path.GetTempPath(), "Backplane") // Uses the folder %temp%/backplane
        let watcher = new FileSystemWatcher()
      
        do
            this.Open(0)  // Use only one stream
            if Directory.Exists basePath then
                Directory.GetFiles(basePath)
                |> Array.iter (File.Delete)
            else Directory.CreateDirectory(basePath) |> ignore

            watcher.Path <- basePath
            watcher.Filter <- "*.txt"
            watcher.IncludeSubdirectories <- false
            watcher.EnableRaisingEvents <- true

            watcher.Created 
            |> Observable.add(this.FileCreated)      // Process messages sent from the backplane to the server
    
        // Send messages from the server to the backplane
        override x.Send(streamIndex:int, messages:IList<Message>) =
        
            System.Threading.Tasks.Task.Factory.StartNew(new Action(fun () ->
                    let msgs = new ScaleoutMessage(messages)
                    let bytes = msgs.ToBytes()
                    let filePath = sprintf "%s\\%d.txt" basePath DateTime.Now.Ticks 
                    File.WriteAllBytes(filePath, bytes) ))

        override x.Dispose(disposing) =
                if disposing then 
                    watcher.Dispose()
                base.Dispose(disposing)
    
        static member Create(resolver:IDependencyResolver, configuration:ScaleoutConfiguration) =
             Lazy(fun () -> new FileSystemBus(resolver, configuration))

        member private x.FileCreated (ev:FileSystemEventArgs) =                 
                    try
                        let bytes = File.ReadAllBytes(ev.FullPath)
                        let scaleoutMessage = ScaleoutMessage.FromBytes(bytes)
                        let fileName = Path.GetFileNameWithoutExtension(ev.Name)
                        let (success, id) = System.UInt64.TryParse(fileName)
                        match success with
                        | true ->   for msg in scaleoutMessage.Messages do 
                                        let msg = List<Message>(Seq.singleton(msg)) :> IList<_>
                                        this.OnReceived(0, id, new ScaleoutMessage( msg ))
                        | _ -> () 
                    with
                    | ex -> let msg = ex.Message
                            let r = msg
                            ()

    //  To inform SignalR that it is the message bus to be used, we would just need to execute the following code during startup:
    //  var bus = new Lazy<FileSystemMessageBus>(
    //              () => new FileSystemMessageBus(
    //                         GlobalHost.DependencyResolver,
    //                         new ScaleoutConfiguration()) );

    //GlobalHost.DependencyResolver.Register(
    //              typeof(IMessageBus),
    //              () => (object)bus.Value );
   
