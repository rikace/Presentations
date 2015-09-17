module mainPipe 

open System
open PipeAsyncServer
open PipeAsyncClient
let myPipe = "mypipe"

let client = new ClientAsyncPipe(myPipe, Option.None, Option.None)
let server = new ServerAsyncPipe(myPipe, None)

server.Connect()
Console.ReadLine() |> ignore
let disp = client.Connect()
   

//    if Array.length argv >0 && argv.[0] = "s" then 
//         PipeServer.ServerSync.StartServerPipe(myPipe)  
//    else
//       PipeClient.ClientSync.StartClientPipe(myPipe)  
//
Console.ReadLine() |> ignore

client.Write(sprintf "%s" (new String('c', 1000)))
server.Write(sprintf "%s" (new String('s', 1000)))

//    [0..5] |> Seq.iter (fun i -> client.Write(sprintf "ciao server bello %d" i))
//    [0..5] |> Seq.iter (fun i -> server.Write(sprintf "ciao client bello %d" i))
   
Console.ReadLine() |> ignore

0 // return an integer exit code
