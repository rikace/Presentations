namespace Easj360FSharp

open System.Net
open System.Net.Sockets

module AsyncNetwork =

    let AsyncConnect (c : TcpClient, ipAddr:IPAddress, port:int) = 
        Async.FromBeginEnd((fun (cb,o)->c.BeginConnect(ipAddr,port,cb,o)), c.EndConnect)
    let bar (ip, port) =
      async
        { 
          let c = new TcpClient()
          printfn "connecting to: %s:%d" ip port
          //c.Connect(IPAddress.Parse(ip), port)
          do! AsyncConnect(c, IPAddress.Parse(ip), port)
          c.Close()
        }
    Async.RunSynchronously(bar("10.00.00.000", 64996), 10)  //the timeout
