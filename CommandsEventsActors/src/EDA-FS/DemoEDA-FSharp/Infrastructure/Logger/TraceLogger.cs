using System;
using System.Diagnostics;

namespace DemoEDAFsharp.Logging
{
    public class TraceLogger : ILog
    {
        private static readonly object Sync = new object();
        private readonly Threshold threshold;

        public TraceLogger(Threshold threshold = Threshold.Info)
        {
            this.threshold = threshold;
        }

        public virtual void Verbose(string message, params object[] values)
        {
            DebugWindow(Threshold.Verbose, message, values);
        }

        public virtual void Debug(string message, params object[] values)
        {
            DebugWindow(Threshold.Debug, message, values);
        }

        public virtual void Info(string message, params object[] values)
        {
            TraceWindow(Threshold.Info, message, values);
        }

        public virtual void Warn(string message, params object[] values)
        {
            TraceWindow(Threshold.Warn, message, values);
        }

        public virtual void Error(string message, params object[] values)
        {
            TraceWindow(Threshold.Error, message, values);
        }

        public virtual void Fatal(string message, params object[] values)
        {
            TraceWindow(Threshold.Fatal, message, values);
        }

        public void Verbose(string message, Exception exception)
        {
            DebugWindow(Threshold.Verbose, message, exception);
        }

        public void Debug(string message, Exception exception)
        {
            DebugWindow(Threshold.Debug, message, exception);
        }

        public void Info(string message, Exception exception)
        {
            TraceWindow(Threshold.Info, message, exception);
        }

        public void Warn(string message, Exception exception)
        {
            TraceWindow(Threshold.Warn, message, exception);
        }

        public void Error(string message, Exception exception)
        {
            TraceWindow(Threshold.Error, message, exception);
        }

        public void Fatal(string message, Exception exception)
        {
            TraceWindow(Threshold.Fatal, message, exception);
        }

        protected virtual void DebugWindow(Threshold severity, string message, params object[] values)
        {
            if (severity < threshold)
                return;

            lock (Sync)
                System.Diagnostics.Debug.WriteLine(severity, message.FormatMessage(values));
        }

        protected virtual void TraceWindow(Threshold severity, string message, params object[] values)
        {
            if (severity < threshold)
                return;

            lock (Sync)
                Trace.WriteLine(severity, message.FormatMessage(values));
        }
    }
}