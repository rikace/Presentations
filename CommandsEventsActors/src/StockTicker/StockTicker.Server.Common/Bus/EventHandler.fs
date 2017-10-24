namespace StockTicker.Events

open System
open System.Collections.Generic
open System.Threading.Tasks
open Events

/// Event broker for event based communication
module EventBus =

    /// Used just to notify others if anyone would be interested
    let public EventPublisher =
        new Microsoft.FSharp.Control.Event<Event>()

    /// Used to subscribe to event changes
    let public Subscribe (eventHandle: Events.Event -> unit) =
        EventPublisher.Publish |> Observable.add(eventHandle)

module EventStorage =

    open EventBus
    open Events
    open System

    type EventStorageMessage =
        | SaveEvent of id:Guid * event:EventDescriptor
        | GetEventsHistory of Guid * AsyncReplyChannel<Event list option>

    /// Custom implementation of in-memory time async event storage. Using message passing.
    type EventStorage() =
        let eventstorage = MailboxProcessor.Start(fun inbox ->
            let rec msgPassing (history:Dictionary<Guid, EventDescriptor list>) = async {
                let! msg = inbox.Receive()
                match msg with
                | SaveEvent(id, event) ->

                            let storeAndPublish evt =
                                EventPublisher.Trigger evt

                            let (success, events) = history.TryGetValue(id)
                            match success with
                            | true -> history.[id] <- (event :: events)
                            | false -> history.Add(id, [event])
                            return! msgPassing(history)

                | GetEventsHistory(id, reply) ->
                            let (success, events) = history.TryGetValue(id)
                            match (success, events) with
                            | (true, ev) -> let events' = events |> List.map (fun i -> i.EventData)
                                            reply.Reply(Some(events'))
                            | (false, _) -> reply.Reply(None)

                            return! msgPassing(history)
                }
            msgPassing (Dictionary<Guid, EventDescriptor list>(HashIdentity.Structural)))

        member x.SaveEvent(id:Guid) (event:EventDescriptor) =
            eventstorage.Post(SaveEvent(id, event))


        member x.GetEventsHistory(id:Guid) f =
                let events = eventstorage.PostAndReply(fun rep -> GetEventsHistory(id,rep))
                match events with
                | Some(evs) ->  evs
                                |> List.rev
                                |> List.iter f
                                |> Some
                | None -> None


