open Owin
open EkonBenefits.FSharp.Dynamic
open Microsoft.AspNet.SignalR
open Microsoft.AspNet.SignalR.Hubs
open Microsoft.Owin.Hosting
open Microsoft.Owin.Cors
open System.Reactive.Linq
open System.Reactive.Concurrency
open System
open System.Threading.Tasks
open AsyncHelper

module MyServer =

    type MyConnection() as this = 
        inherit PersistentConnection()
        
        override x.OnConnected(req,id) =
            this.Connection.Send(id, "Welcome!") |> ignore
            base.OnConnected(req,id)
       
        override x.OnReceived(req,id,data) =
            printfn "%s sent meesage %s" id data
            this.Connection.Broadcast(data) |> ignore
            base.OnReceived(req,id,data)

    let sendAll msg = 
            GlobalHost.ConnectionManager.GetConnectionContext<MyConnection>().Connection.Broadcast(msg) 
                |> Async.awaitPlainTask |> ignore

    let SignalRCommunicationSendPings() =
        //Send first message
        sendAll("Hello world!")

        //Send ping on every 5 seconds
        let pings = Observable.Interval(TimeSpan.FromSeconds(5.0), Scheduler.Default)
        pings.Subscribe(fun s -> sendAll("ping!")) |> ignore

    open Microsoft.Owin.Hosting
    let config = new HubConfiguration(EnableDetailedErrors = true)

    type MyWebStartup() =
        member x.Configuration(app:Owin.IAppBuilder) =
            Owin.OwinExtensions.MapSignalR<MyConnection>(app, "/signalrConn") |> ignore
            ()

    [<assembly: Microsoft.Owin.OwinStartup(typeof<MyWebStartup>)>]
    do()

open MyServer

[<EntryPoint>]
let main argv = 
   Console.Title <- "SignalR Server"
   let hostUrl = "http://localhost:9099"

   let startup (a:IAppBuilder) =
       a.UseCors(CorsOptions.AllowAll) |> ignore
       a.MapSignalR() |> ignore
   
   use app =  WebApp.Start<MyServer.MyWebStartup>(hostUrl) 
   Console.WriteLine("Server running on "+ hostUrl)
   Console.WriteLine()

   Console.ReadLine() |> ignore

   SignalRCommunicationSendPings()

   Console.ReadLine() |> ignore
   0
