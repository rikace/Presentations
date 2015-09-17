module MessageBus

open System
open System.Threading
open System.Threading.Tasks
open System.Collections.Concurrent
open System.Linq
open System.Reactive.Subjects


type IEventHandler<'TEvent> = 
    abstract member HandleEvent : 'TEvent -> unit

type IRequestHandler<'TRequest, 'TResponse> = 
    abstract member HandleRequest : 'TRequest -> Task<'TResponse>

type EventPublisher<'TEvent> = System.Action<'TEvent>

type RequestSender<'TRequest, 'TResponse> = System.Func<'TRequest, Task<'TResponse>>

type MessageProcessingMode = 
    | Sequential = 0
    | Concurrent = 1

type ExceptionMessage(ex : exn, msg : obj, service : obj) =
    member x.Exception = ex
    member x.FaultedMessage = msg
    member x.FaultedService = service

type IMessageBus = 
    abstract member RegisterEventHandler: IEventHandler<'TEvent> -> MessageProcessingMode -> unit
    abstract member RegisterRequestHandler: IRequestHandler<'TRequest, 'TResponse> -> MessageProcessingMode -> unit
    abstract member ToEventPublisher: unit -> EventPublisher<'TEvent>
    abstract member ToRequestSender: unit -> RequestSender<'TRequest, 'TResponse>
    abstract member ExceptionPublisher: EventPublisher<ExceptionMessage>

type IMessageQueue = 
    abstract member Publish : 'TEvent -> EventPublisher<ExceptionMessage> -> unit
    abstract member Send : 'TRequest -> EventPublisher<ExceptionMessage> -> Task<'TResponse>
    abstract member Service : obj



