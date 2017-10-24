using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Reactive.Linq;
using System.Threading;
using System.Threading.Tasks;
using Common;
using Common.Framework;

namespace DemoEDAFSharp.Framework
{
    public sealed class Bus : ICommandDispatcher, IEventPublisher, ISubscriber
    {
        private readonly Dictionary<Type, List<Action<IMessage>>> _actions =
            new Dictionary<Type, List<Action<IMessage>>>();

        public void Dispatch<TCommand>(TCommand command) where TCommand : ICommand
        {
            List<Action<IMessage>> handlers;
            if (_actions.TryGetValue(typeof(TCommand), out handlers))
            {
                handlers[0](command);
            }
            else
            {
                throw new InvalidOperationException("no handler registered");
            }
            Thread.Sleep(500);
        }

        public void Publish<TEvent>(TEvent @event) where TEvent : IEvent
        {
            List<Action<IMessage>> handlers;
            if (!_actions.TryGetValue(@event.GetType(), out handlers)) return;
            foreach (var handler in handlers)
            {
                Action<IMessage> handler1 = handler;
                // handler1(@event);
                Task.Factory.StartNew(x => handler1(@event), handler1);
            }
        }

        public void RegisterHandler<T>(Action<T> handler) where T : IMessage
        {
            List<Action<IMessage>> handlers;
            if (!_actions.TryGetValue(typeof(T), out handlers))
            {
                handlers = new List<Action<IMessage>>();
                _actions.Add(typeof(T), handlers);
            }
            handlers.Add(ActionCastHelper.CastArgument<IMessage, T>(x => handler(x)));
        }


    }
}