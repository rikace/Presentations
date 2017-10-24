namespace EDAFSharp

open System
open Domain.Entities
open System.Threading
open System.Threading.Tasks
open System.Data.Linq

open System.Collections.Generic
open System.Data.Entity
open Microsoft.FSharp.Data.TypeProviders
open FSharp.Data.SqlClient
open FSharp.Interop.Dynamic
open Microsoft.AspNet.SignalR
open DemoEDAFSharp.Infrastructure
open System.Dynamic
open System.Collections.Concurrent
open Domain.DataAccess

module MessageBus =

    type CommandHandler =
        abstract member Execute : Command -> unit
    // Discrimination Union
    (* DU is a data type with a finite number of
       alternative representations *)
    and Command =
    | AddProductCommand         of Guid * Product * int
    | RemoveProductCommand      of Guid * Product
    | SubmitOrderCommand        of Guid * Order
    | EmptyCardCommand          of Guid
    with
        member this.ToEvents () =
            match this with
            | AddProductCommand(id, product, quantity) -> [ProductAddedEvent(product, id)]
            | RemoveProductCommand(id, product) -> [ProductRemovedEvent(product, id)]
            | SubmitOrderCommand(id, order) -> [OrderSubmittedEvent(order, id)]
            | EmptyCardCommand(id) -> [EmptyCardEvent(id)]
        member this.Execute() =
              let op1 = match this with
                        | AddProductCommand(id, product, quantity) -> CommandHandlers.addProductCommand(string(id), product, quantity)
                        | RemoveProductCommand(id, product) -> CommandHandlers.removeProductCommand(string(id), product)
                        | SubmitOrderCommand(id, order) -> CommandHandlers.submitOrderCommand(string(id), order)
                        | EmptyCardCommand(id) ->   CommandHandlers.emptyCardCommand(string(id))
              // Event Store
              (* Event Store gives you a straightforward persistence engine for
                 applications using event-sourcing, as well as being great for
                 storing time-series data. *)
              let op2 = async { this.ToEvents() |> List.iter(EventStore().SaveEvents) }
              [op1;op2]
    and IEventHandler =
        abstract member Handle : Event -> unit
    and Event =
    | OrderSubmittedEvent       of Order * Guid
    | ProductAddedEvent         of Product * Guid
    | ProductRemovedEvent       of Product * Guid
    | EmptyCardEvent            of Guid
    // Event Publisher
    and EventPublisher() =
        let agentEvent = Agent.Start(fun inbox ->
                 let rec loop() = async {
                        let! eventMsg = inbox.Receive()
                        // Pattern Matching
                        (* PM are rules to transform data, compare data and decompose data
                           "like switch statemnts on steroids" *)
                        match eventMsg with
                        | OrderSubmittedEvent(order,id) -> EventHandlers.orderSubmittedEvent |> List.iter(fun h -> h(order, id))
                        | ProductAddedEvent(product, id) -> EventHandlers.productAddedEvent |> List.iter(fun h -> h(product, id))
                        | ProductRemovedEvent(product, id) -> EventHandlers.productRemoveEvent |> List.iter(fun h -> h(product, id))
                        | EmptyCardEvent(id) -> EventHandlers.emptyCardEvent |> List.iter(fun h -> h(id)) }
                 loop())
        member this.Publish(event:Event) =  agentEvent.Post(event)
        member this.Publish(id:Guid, event:Event) = agentEvent.Post(event)
    and EventHandlers() =
        static member orderSubmittedEvent = [
                                    (fun (order:Order, id:Guid) ->
                                                let orderSubmittedEvent' = OrderSubmittedEvent(order, id)
                                                let ev = { new IEventHandler with
                                                            member x.Handle(ev) =  (*Semd Email*)
                                                                printfn "Event send email for id %A" id;  ()}
                                                async { ev.Handle(orderSubmittedEvent') } |> Async.Start )

                                    (fun (order:Order, id:Guid) ->
                                                let notifyWareHouse' = OrderSubmittedEvent(order, id)
                                                let ev = { new IEventHandler with
                                                        member x.Handle(ev) =  (*notify WareHouse *)
                                                            printfn "Event Notify Werehouse for id %A" id;  ()}
                                                async { ev.Handle(notifyWareHouse') } |> Async.Start  )
                              ]
        static member productAddedEvent = [
                                    (fun (product:Product, id:Guid) ->
                                                let orderSubmittedEvent' = ProductAddedEvent(product, id)
                                                let ev = { new IEventHandler with
                                                        member x.Handle(ev) =
                                                            let ctx = GlobalHost.ConnectionManager.GetHubContext<SignalRHub>()
                                                            ctx.Clients.All?broadcastMessage(String.Format("Product {0} has been added succesfully", product.Name))
                                                            printfn "Event product added for id %A" id;  ()}
                                                async { ev.Handle(orderSubmittedEvent') } |> Async.Start  )

                                    (fun (product:Product, id:Guid) ->
                                                let orderSubmittedEvent' = ProductAddedEvent(product, id)
                                                let ev = { new IEventHandler with
                                                        member x.Handle(ev) =  (*Log Product Added*) printfn "Event product added for id %A" id;  ()}
                                                async { ev.Handle(orderSubmittedEvent') } |> Async.Start )
                              ]
        static member productRemoveEvent = [
                                    (fun (product:Product, id:Guid) ->
                                                let productRemovedEvent' = ProductRemovedEvent(product, id)
                                                let ev = { new IEventHandler with
                                                        member x.Handle(ev) =
                                                            let ctx = GlobalHost.ConnectionManager.GetHubContext<SignalRHub>()
                                                            ctx.Clients.All?broadcastMessage(String.Format("Product {0} has been removed succesfully", product.Name))
                                                            printfn "Event product removed for id %A" id;  ()}
                                                async { ev.Handle(productRemovedEvent') } |> Async.Start  )

                                    (fun (product:Product, id:Guid) ->
                                                let productRemovedEvent' = ProductRemovedEvent(product, id)
                                                let ev = { new IEventHandler with
                                                        member x.Handle(ev) =  (*Log Product Remove*) printfn "Event product removed for id %A" id;  ()}
                                                async { ev.Handle(productRemovedEvent') } |> Async.Start )
                              ]
        static member emptyCardEvent = [(fun (id:Guid) ->
                                                let emptyCardEvent' = EmptyCardEvent(id)
                                                let ev = { new IEventHandler with
                                                        member x.Handle(ev) =
                                                            let ctx = GlobalHost.ConnectionManager.GetHubContext<SignalRHub>()
                                                            ctx.Clients.All?broadcastMessage("The Cart is empty")
                                                            printfn "Event product removed for id %A" id;  ()}
                                                async { ev.Handle(emptyCardEvent') } |> Async.Start  )]
    and FSharpBus() as bus =
        let agentCommands = Agent<CommandReplay>.Start(fun inbox ->
            let rec loop() = async {
                let! msg = inbox.Receive()
                match msg with
                | CommandReplay(cmd, replay) -> //let AddProductCommand(id, product , quantity) = cmd
                                                cmd.Execute() @ [Async.Sleep 500]
                                                |> Async.Parallel
                                                |> Async.RunSynchronously |> ignore
                                                replay.Reply()
                return! loop() }
            loop() )
        member bus.DispacthCommand(cmd:Command) =
            let task = agentCommands.PostAndAsyncReply(fun replay -> CommandReplay(cmd, replay)) |> Async.StartAsTask
            task.Wait()
    and CommandReplay =
    | CommandReplay of Command * AsyncReplyChannel<unit>
    // Record Type
    and EventDescriptor = {id:Guid; eventData:Event}
    and EventStore() =
        let agentEventStore = Agent.Start(fun inbox ->
            let events = new Dictionary<Guid, EventDescriptor list>()
            let rec loop(events:Dictionary<Guid, EventDescriptor list>) = async {
                let!message = inbox.Receive()
                match message with
                | SaveEvent(id, event) ->   let ok, evDescriptor = events.TryGetValue(id)
                                            match ok with
                                            | true ->  let eventDesList = events.[id] @ [{id = id; eventData = event}]
                                                       events.[id] <- eventDesList
                                            | false -> let eventDesList = [{id = id; eventData = event}]
                                                       events.Add(id, eventDesList)
                                            events.[id] |> List.map (fun e -> e.eventData) |> List.iter (EventPublisher().Publish)
                | GetEvents(id, replay) ->  let ok, evDescriptor = events.TryGetValue(id)
                                            if ok then
                                                replay.Reply(evDescriptor)
                return! loop(events) }
            loop(events) )

        member x.SaveEvents(event:Event, ?id:Guid) =
            let id' = defaultArg id (Guid.NewGuid())
            agentEventStore.Post(SaveEvent(id', event))
        member x.GetEvents(id) =
            agentEventStore.PostAndAsyncReply(fun replay -> GetEvents(id, replay))
    and EventStoreCommand =
    | SaveEvent of Guid * Event
    | GetEvents of Guid * AsyncReplyChannel<EventDescriptor list>


open System.Reactive
open System.Reactive.Subjects

type SubjectMessaage<'a> =
    | Subscribe of IObserver<'a> // async reply
    | Unsubscribe of IObserver<'a>
    | OnNext of 'a
    | OnError of System.Exception
    | OnCompleted

 type internal ThrottlingAgentMessage =
    | Completed
    | Work of Async<unit>


 type FsharpSubject<'a>(?limit:int) =
    let limit = defaultArg limit 1
    let throttlingAgent = MailboxProcessor.Start(fun agent ->
            let rec waiting () =
              agent.Scan(function
                | Completed -> Some(working (limit - 1))
                | _ -> None)
            and working count = async {
              let! msg = agent.Receive()
              match msg with
              | Completed -> return! working (count - 1)
              | Work work ->  async { try do! work
                                      finally agent.Post(Completed) } |> Async.Start
                              if count < limit then return! working (count + 1)
                              else return! waiting () }
            working 0)

    let agent = MailboxProcessor<_>.Start(fun inbox ->
        let notifyObserver linkList f =  throttlingAgent.Post(Work (async { linkList |> Seq.iter f }))
        let rec loop (linkList:System.Collections.Generic.List<IObserver<'a>>) = async {
            let! msg = inbox.Receive()
            match msg with
            | Subscribe(observer)   ->  if not( linkList.Contains(observer) ) then linkList.Add observer
            | Unsubscribe(observer) ->  if linkList.Contains(observer) then ignore( linkList.Remove(observer) )
            | OnNext(item)      ->  notifyObserver linkList (fun observer -> observer.OnNext(item))
            | OnError(error)    ->  notifyObserver linkList (fun observer -> observer.OnError(error))
            | OnCompleted       ->  notifyObserver linkList (fun observer -> observer.OnCompleted())
            return! loop(linkList) }
        loop (System.Collections.Generic.List<IObserver<'a>>()) )

    interface System.Reactive.Subjects.ISubject<'a> with
        member this.OnNext(value:'a) =
            agent.Post(OnNext(value))

        member this.OnError(error:System.Exception) =
            agent.Post(OnError(error))

        member this.OnCompleted() =
            agent.Post(OnCompleted)

        member this.Subscribe(observer:IObserver<'a>) =
            agent.Post(Subscribe(observer))
            { new System.IDisposable with
                member this.Dispose() =
                    let unsubscribe = agent.Post(Unsubscribe(observer))
                    unsubscribe }