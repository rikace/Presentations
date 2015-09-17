namespace Easj360FSharp

module AsyncHttpServer =

    open System
    open System.Net

    type HttpListener with
        static member Run (url:string,handler: (HttpListenerRequest -> HttpListenerResponse -> Async<unit>)) = 
            let listener = new HttpListener()
            listener.Prefixes.Add url
            listener.Start()
            let asynctask = Async.FromBeginEnd(listener.BeginGetContext,listener.EndGetContext)
            async {
                while true do 
                    let! context = asynctask
                    Async.Start (handler context.Request context.Response)
            } |> Async.Start 
            listener

    HttpListener.Run("http://*:80/App/",(fun req resp -> 
            async {
                let out = Text.Encoding.ASCII.GetBytes "hello world"
                resp.OutputStream.Write(out,0,out.Length)
                resp.OutputStream.Close()
            }
        )) |> ignore

    Console.Read () |> ignore