namespace ChatSignalRServer

open Microsoft.AspNet.SignalR
open Microsoft.AspNet.SignalR.Hubs
open System.Threading
open System.Threading.Tasks
open System
open Common

module ChatServer =

    type Agent<'T> = MailboxProcessor<'T>
    
    type IChatHubClient =
        abstract member BroadcastMessage : string -> unit
               
    [<HubName("chatHub")>]
    type ChatRoomHub() = 
        inherit Hub<IChatHubClient>()

        // static because the object is created for each new connection
        static let users = new System.Collections.Concurrent.ConcurrentDictionary<string, string>(HashIdentity.Structural)
        static let userCount = ref 0

        // agent isolation
        // This aspect is of vital importance in all SignalR applications.
        // We are building multiuser systems where we will likely have a high degree of concurrency, 
        // so it is necessary to take precautions to prevent simultaneous access of these members.
        static let chatAgent = Agent<ChatCommand>.Start(fun inbox ->            
            let rec loop (users:Map<string,string>) = async {
                let! message = inbox.Receive()
                match message with
                | JoinChat(id, userName, ctx) ->  
                        let user = users.TryFind(id)
                        match user with
                        | Some _ -> return! loop users 
                        | None -> 
                                let users' = users.Add(id, userName)                                
                                ctx.Clients.All.BroadcastMessage(sprintf "%s has joined the Chat" userName)
                                return! loop users'                              
                | LeaveChat(id, ctx) ->  
                        let user = users.TryFind(id)
                        match user with
                        | Some u -> 
                              let users' = users.Remove(id)
                              ctx.Clients.All.BroadcastMessage(sprintf "%s has left the Chat" u)
                              return! loop users' 
                        | None -> return! loop users 
                | SendMessage(id, message, ctx) -> 
                        let user = users.TryFind(id)
                        match user with
                        | Some u -> ctx.Clients.All.BroadcastMessage(sprintf "%s said : %s" u message)
                                    return! loop users
                        | None -> return! loop users 

                        return! loop users }
            loop (Map.empty<string,string>) )
        


        member this.SendMessage(text : string) =
            let id = this.Context.ConnectionId
            chatAgent.Post(SendMessage(id, text, this))

        member this.JoinChat(userName:string) =
            let id = this.Context.ConnectionId
            chatAgent.Post(JoinChat(id, userName, this))

        member this.LeaveChat(userName:string) =
            let id = this.Context.ConnectionId
            chatAgent.Post(LeaveChat(id, this))

        override this.OnConnected() : Task=
            ignore <| Interlocked.Increment(userCount) 
            base.OnConnected()

        override this.OnDisconnected(stopCalled:bool) : Task=
            ignore <| Interlocked.Decrement(userCount) 
            base.OnDisconnected(stopCalled)

        override this.OnReconnected() : Task=
            base.OnReconnected()

    and ChatCommand =
        | JoinChat of string * string * ChatRoomHub
        | LeaveChat of string * ChatRoomHub
        | SendMessage of string * string * ChatRoomHub
