using System;
using System.Collections.Generic;
using System.Reactive.Concurrency;
using System.Reactive.Linq;
using System.Reactive.Subjects;
using System.Threading;
using Common;
using Common.Framework;

namespace DemoEDAFSharp.Framework
{
    public sealed class BusObservable : ICommandDispatcher, IEventPublisher, ISubscriber
    {

        public void Dispatch<TCommand>(TCommand command) where TCommand : ICommand
        {
            ISubject<IMessage> handlers;
            if (_actions.TryGetValue(typeof(TCommand), out handlers))
            {
                handlers.OnNext(command);
            }
            else
            {
                throw new InvalidOperationException("no handler registered");
            }
            Thread.Sleep(500);
        }

        public void Publish<TEvent>(TEvent @event) where TEvent : IEvent
        {
            ISubject<IMessage> handlers;
            if (!_actions.TryGetValue(@event.GetType(), out handlers))
                return;

            handlers.OnNext(@event);
        }

        private readonly Dictionary<Type, ISubject<IMessage>> _actions = new Dictionary<Type, ISubject<IMessage>>();

        public void RegisterHandler<T>(Action<T> handler) where T : IMessage
        {
            ISubject<IMessage> handlers;
            if (!_actions.TryGetValue(typeof(T), out handlers))
            {
                handlers = new Subject<IMessage>();
                _actions.Add(typeof(T), handlers);
            }

            Subject.Synchronize(handlers).SubscribeOn(Scheduler.Default)
                .Subscribe(ActionCastHelper.CastArgument<IMessage, T>(x => handler(x)));
        }
    }
}