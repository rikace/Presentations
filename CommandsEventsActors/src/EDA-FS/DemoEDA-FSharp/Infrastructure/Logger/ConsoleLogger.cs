using System;
using System.ComponentModel.Composition;

namespace DemoEDAFsharp.Logging
{
    [Export(typeof (ILog))]
    [PartCreationPolicy(CreationPolicy.NonShared)]
    public class ConsoleLogger : ILog
    {
        private static readonly object Sync = new object();
        private readonly ConsoleColor originalColor = Console.ForegroundColor;
        private readonly Threshold threshold;

        public ConsoleLogger(Threshold threshold = Threshold.Info)
        {
            this.threshold = threshold;
        }

        public virtual void Verbose(string message, params object[] values)
        {
            Log(ConsoleColor.DarkGreen, Threshold.Verbose, message, values);
        }

        public virtual void Debug(string message, params object[] values)
        {
            Log(ConsoleColor.Green, Threshold.Debug, message, values);
        }

        public virtual void Info(string message, params object[] values)
        {
            Log(ConsoleColor.White, Threshold.Info, message, values);
        }

        public virtual void Warn(string message, params object[] values)
        {
            Log(ConsoleColor.Yellow, Threshold.Warn, message, values);
        }

        public virtual void Error(string message, params object[] values)
        {
            Log(ConsoleColor.DarkRed, Threshold.Error, message, values);
        }

        public virtual void Fatal(string message, params object[] values)
        {
            Log(ConsoleColor.Red, Threshold.Fatal, message, values);
        }

        public void Verbose(string message, Exception exception)
        {
            Log(ConsoleColor.DarkGreen, Threshold.Verbose, message, exception);
        }

        public void Debug(string message, Exception exception)
        {
            Log(ConsoleColor.Green, Threshold.Debug, message, exception);
        }

        public void Info(string message, Exception exception)
        {
            Log(ConsoleColor.White, Threshold.Info, message, exception);
        }

        public void Warn(string message, Exception exception)
        {
            Log(ConsoleColor.Yellow, Threshold.Warn, message, exception);
        }

        public void Error(string message, Exception exception)
        {
            Log(ConsoleColor.DarkRed, Threshold.Error, message, exception);
        }

        public void Fatal(string message, Exception exception)
        {
            Log(ConsoleColor.Red, Threshold.Fatal, message, exception);
        }

        protected virtual void Log(ConsoleColor color, Threshold severity, string message, params object[] values)
        {
            if (severity < threshold)
                return;

            lock (Sync)
            {
                Console.ForegroundColor = color;

                Console.WriteLine(message.FormatMessage(values));
                Console.ForegroundColor = originalColor;
            }
        }
    }
}