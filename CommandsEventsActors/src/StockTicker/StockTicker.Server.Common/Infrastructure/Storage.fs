namespace StockTicker

open System


module InMemory =
    type Agent<'T> = MailboxProcessor<'T>

    type StorageOperation<'a when 'a :> obj> =

        | GetAll of channel: AsyncReplyChannel<'a[]>
        | Put of item: 'a * channel: AsyncReplyChannel<int>
        | Clear
        | Get of index: int * channel: AsyncReplyChannel<'a option>
        | Update of index: int * patch: 'a * channel: AsyncReplyChannel<'a option>
        | Remove of index: int * channel: AsyncReplyChannel<bool>


    type Store<'a>() =
        let getItem index (items: 'a[]) =
            match items with
            | items when items.Length > index ->
                            let item = items.[index]
                            Some item
            | _ -> None


        let agent =
            Agent<_>.Start(fun inbox ->
                let rec loop items =
                    async {
                        let! msg = inbox.Receive()
                        match msg with
                        | GetAll ch -> ch.Reply items
                                       return! loop items
                        | Put(item, ch) ->
                            let index = items.Length
                            ch.Reply index
                            return! loop (Array.append items [| item |])
                        | Clear -> return! loop [||]
                        | Get(index, ch) ->
                            let item = getItem index items
                            ch.Reply item
                            return! loop items
                        | Update(index, item, ch) ->
                            let item =
                                match getItem index items with
                                | Some temp ->
                                    items.[index] <- item
                                    Some item
                                | None -> None
                            ch.Reply item
                            return! loop items
                        | Remove(index, ch) ->
                            let result =
                                match getItem index items with
                                | Some _ ->
                                        items
                                        |> Array.mapi( fun i el -> (i <> index, el))
                                        |> Array.filter fst
                                        |> Array.map snd
                                | None -> items
                            ch.Reply (result.Length < items.Length)
                            return! loop result
                    }
                loop [||])


        member __.GetAll() = agent.PostAndAsyncReply GetAll
        member __.Put(item) = agent.PostAndAsyncReply(fun ch -> Put(item, ch))
        member __.Clear() = agent.Post Clear
        member __.Get(index) = agent.PostAndAsyncReply(fun ch -> Get(index, ch))
        member __.Update(index, item) = agent.PostAndAsyncReply(fun ch -> Update(index, item, ch))
        member __.Remove(index) = agent.PostAndAsyncReply(fun ch -> Remove(index, ch))