using System;
using System.Threading;
using Common.Framework;
using DemoEDAFSharp.Events;
using DemoEDAFSharp.Framework;
using DemoEDAFSharp.Framework.Events;
using Domain;
using EDAFSharp;

namespace DemoEDAFSharp
{
    public sealed class ServiceLocator
    {
        //private static readonly Lazy<Bus> commandBus = new Lazy<Bus>(() => new Bus(),
        //    LazyThreadSafetyMode.PublicationOnly);

        private static readonly Lazy<BusFSharpObservable> commandBus = new Lazy<BusFSharpObservable>(() => new BusFSharpObservable(),
         LazyThreadSafetyMode.PublicationOnly);

        private static readonly Lazy<IEventStore<Event>> eventStore =
            new Lazy<IEventStore<Event>>(() => new EventStore(EventBus),
                LazyThreadSafetyMode.PublicationOnly);

        private static readonly Lazy<MessageBus.FSharpBus> fsharpBus =
            new Lazy<MessageBus.FSharpBus>(() => new MessageBus.FSharpBus(),
                LazyThreadSafetyMode.PublicationOnly);


        static ServiceLocator()
        {
        }

        public static ICommandDispatcher CommandBus
        {
            get { return commandBus.Value; }
        }

        public static ISubscriber Subscriber
        {
            get { return commandBus.Value; }
        }

        public static IEventPublisher EventBus
        {
            get { return commandBus.Value; }
        }

        public static IEventStore<Event> EventStore
        {
            get { return eventStore.Value; }
        }

        public static MessageBus.FSharpBus FSharpBus
        {
            get { return fsharpBus.Value; }
        }
    }
}