open System
open Microsoft.AspNet.SignalR.Client
open AsyncHelper

[<EntryPoint>]
let main argv = 

    Console.Title <- "SignalR Console Client"

    let connection = new Connection("http://localhost:9099/signalrConn")

    connection.AsObservable()
    |> Observable.add(fun s -> printfn "message Received : %s\n" s)

    connection.Start().Wait()

    let rec keepSendingMessages message = 
        match message with
        | m when not <| String.IsNullOrWhiteSpace(m) -> 
                         connection.Send(m).Wait()
                         printf "Message Input : "
                         keepSendingMessages (Console.ReadLine())
        | _ -> ()
    
    printf "Message Input : "  
    keepSendingMessages (Console.ReadLine())


    0