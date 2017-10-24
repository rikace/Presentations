namespace StockTicker

module CacheAgent =
   // Discriminated Union of possible incoming messages
    type private Message =
        | Get of string * AsyncReplyChannel<obj option>
        | Set of string * obj

    // Core agent
    let private agent = MailboxProcessor.Start(fun inbox ->
        let rec loop(cacheMap:Map<string, obj>) =
            async {
                let! message = inbox.Receive()
                match message with
                | Get(key, replyChannel) ->
                    Map.tryFind key cacheMap |> replyChannel.Reply
                | Set(key, data) ->
                    do! loop( (key, data) |> cacheMap.Add)
                do! loop cacheMap
            }
        loop Map.empty)

    // Public function that retrieves the data from cache as an Option
    let get<'a> key =
        agent.PostAndReply(fun reply -> Message.Get(key, reply))
        |> function
           | Some v -> v :?> 'a |> Some
           | None -> None

    // Public function that sets the cached data
    let set key value =
        Message.Set(key, value) |> agent.Post


