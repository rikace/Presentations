using System.Diagnostics;
using DemoEDA.Infrastructure;
using DemoEDAFsharp.Logging;

namespace DemoEDA.Logger
{
    public class OutputLogger : ILog
    {
        private static readonly object Sync = new object();

        public virtual void Verbose(string message, params object[] values)
        {
            DebugWindow("Verbose", message, values);
        }

        public virtual void Debug(string message, params object[] values)
        {
            DebugWindow("Debug", message, values);
        }

        public virtual void Info(string message, params object[] values)
        {
            TraceWindow("Info", message, values);
        }

        public virtual void Warn(string message, params object[] values)
        {
            TraceWindow("Warn", message, values);
        }

        public virtual void Error(string message, params object[] values)
        {
            TraceWindow("Error", message, values);
        }

        public virtual void Fatal(string message, params object[] values)
        {
            TraceWindow("Fatal", message, values);
        }

        private void DebugWindow(string category, string message, params object[] values)
        {
            lock (Sync)
                System.Diagnostics.Debug.WriteLine(category, StringEx.FormatMessage(message, values));
        }

        private void TraceWindow(string category, string message, params object[] values)
        {
            lock (Sync)
                Trace.WriteLine(category, StringEx.FormatMessage(message, values));
        }
    }
}