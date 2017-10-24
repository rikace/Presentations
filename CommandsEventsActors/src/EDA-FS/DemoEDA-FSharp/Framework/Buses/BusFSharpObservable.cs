using System;
using System.Collections.Generic;
using System.Reactive.Subjects;
using System.Threading;
using Common;
using Common.Framework;
using EDAFSharp;
using Microsoft.FSharp.Core;

namespace DemoEDAFSharp.Framework
{
    public sealed class BusFSharpObservable : ICommandDispatcher, IEventPublisher, ISubscriber
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
            if (_actions.TryGetValue(@event.GetType(), out handlers))
                handlers.OnNext(@event);
        }

        private readonly Dictionary<Type, ISubject<IMessage>> _actions = new Dictionary<Type, ISubject<IMessage>>();

        public void RegisterHandler<T>(Action<T> handler) where T : IMessage
        {
            ISubject<IMessage> handlers;
            if (!_actions.TryGetValue(typeof(T), out handlers))
            {
                handlers = new FsharpSubject<IMessage>(new FSharpOption<int>(4));
                _actions.Add(typeof(T), handlers);
            }

            handlers.Subscribe(ActionCastHelper.CastArgument<IMessage, T>(c => handler(c)));
        }
    }
}