using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;

namespace DemoEDAFSharp.Infrastructure
{
    public static class LoggerEx
    {
        public static Action<TMessage> WrapLogger<TMessage>(Action<TMessage> action)
        {
            return new LoggingExecutor<TMessage>(action).Handle;
        }
    }
}