using System;
using System.ComponentModel.Composition;

namespace DemoEDAFsharp.Logging
{
    [Export]
    public static class LogFactory
    {
        static LogFactory()
        {
            LogWith(new NullLogger());
        }

        public static ILog Logger { get; private set; }

        public static void LogWith(ILog logger)
        {
            Logger = logger ?? new NullLogger();
        }

        private class NullLogger : ILog
        {
            public void Verbose(string message, params object[] values)
            {
            }

            public void Debug(string message, params object[] values)
            {
            }

            public void Info(string message, params object[] values)
            {
            }

            public void Warn(string message, params object[] values)
            {
            }

            public void Error(string message, params object[] values)
            {
            }

            public void Fatal(string message, params object[] values)
            {
            }

            public void Verbose(string message, Exception exception)
            {
            }

            public void Debug(string message, Exception exception)
            {
            }

            public void Info(string message, Exception exception)
            {
            }

            public void Warn(string message, Exception exception)
            {
            }

            public void Error(string message, Exception exception)
            {
            }

            public void Fatal(string message, Exception exception)
            {
            }
        }
    }
}