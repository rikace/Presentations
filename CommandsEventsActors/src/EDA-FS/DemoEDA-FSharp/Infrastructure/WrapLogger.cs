using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using DemoEDAFsharp.Logging;

namespace DemoEDAFSharp.Infrastructure
{
    internal class LoggingExecutor<TMessage>
    {
        private readonly Action<TMessage> _action;

        public LoggingExecutor(Action<TMessage> action)
        {
            _action = action;
        }

        public void Handle(TMessage message)
        {
            LogFactory.Logger.Info("Log: Action type {0} - Time ", typeof(TMessage), DateTime.Now);
            Console.WriteLine("Log: Action type {0} - Time ", typeof(TMessage), DateTime.Now);
            _action(message);
        }
    }
}