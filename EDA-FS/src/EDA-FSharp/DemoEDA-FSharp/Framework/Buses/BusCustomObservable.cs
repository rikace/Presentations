using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Reactive.Subjects;
using System.Threading;
using System.Threading.Tasks;
using Common;
using Common.Framework;
using WebGrease.Css.Extensions;

namespace DemoEDAFSharp.Framework
{
    public sealed class BusCustomObservable : ICommandDispatcher, IEventPublisher, ISubscriber
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
            if (!_actions.TryGetValue(typeof (T), out handlers))
            {
                handlers = new EDASubject<IMessage>();
                _actions.Add(typeof (T), handlers);
            }

            handlers.Subscribe(ActionCastHelper.CastArgument<IMessage, T>(c => handler(c)));
        }
    }

    internal class EDASubject<T> : ISubject<T>
    {
        private ConcurrentBag<IObserver<T>> observers;

        public EDASubject()
        {
            observers = new ConcurrentBag<IObserver<T>>();
        }

        public void OnCompleted()
        {
            Task.Factory.StartNew(() => observers.ForEach(o => o.OnCompleted()));
        }

        public void OnError(Exception error)
        {
            Task.Factory.StartNew(() => observers.ForEach(o => o.OnError(error)));
        }

        public void OnNext(T value)
        {
            Task.Factory.StartNew(() => observers.ForEach(o => o.OnNext(value)));
        }

        public IDisposable Subscribe(IObserver<T> observer)
        {
            if (!observers.Contains(observer))
                observers.Add(observer);
            return new Unsubscriber<T>(observers, observer);
        }

        internal class Unsubscriber<T> : IDisposable
        {
            private ConcurrentBag<IObserver<T>> _observers;
            private IObserver<T> _observer;
            internal Unsubscriber(ConcurrentBag<IObserver<T>> observers, IObserver<T> observer)
            {
                this._observers = observers;
                this._observer = observer;
            }
            public void Dispose()
            {
                if (_observers.Contains(_observer))
                    _observers.TryTake(out _observer);
            }
        }
    }
}