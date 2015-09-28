[<ReflectedDefinition>]
module ChatSignalRClient 

open FunScript.TypeScript
open FunScript

open SignalRProvider

let signalR = Globals.Dollar.signalR
let j (s: string) = Globals.Dollar.Invoke(s)
let log = Globals.console.log

let serverHub = new Hubs.chatHub(signalR.hub)

let onstart () =     
    j("#joinChatBtn").click (fun _ -> 
        let userName = string (unbox (j("#userName")._val()))
        serverHub.JoinChat(userName) |> ignore
        j("#chatDiv").show() |> ignore
        j("#userName").hide() |> ignore
        j("#joinChatBtn").hide() |> ignore
        j("#leaveChatBtn").show() |> ignore
      
        j("#leaveChatBtn").click (fun _ -> 
            serverHub.LeaveChat(userName) |> ignore
            j("#chatDiv").hide() |> ignore
            j("#userName").show() |> ignore
            j("#leaveChatBtn").hide() |> ignore
            j("#joinChatBtn").show() |> ignore
            new obj()
            ) |> ignore
        new obj() ) |> ignore

    log "##Started!##"

    j("#submit").click (fun _ -> 
        serverHub.SendMessage (j("#source")._val() :?> string) |> ignore
        new obj()
        )
    |> ignore
    log "##Sent MEssage!##"

let printResult (value : string) =
    //sprintf "<p>%s</p>" value
    "<p>"+ value + "" + "</p>"
    |> j("#results").append 
    |> ignore


let main() = 
    Globals.console.log("##Starting:## ")
    signalR.hub.url <- "http://localhost:48430/signalr/hubs"

    let client = Hubs.ChatHubClient()            
    client.BroadcastMessage <- (fun msg -> printResult msg)
    client.Register(signalR.hub)
    
    signalR.hub.start onstart

type Wrapper() =
    member this.GenerateScript() = Compiler.compileWithoutReturn <@ main() @>