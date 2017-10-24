using System;
using System.Globalization;
using System.Threading;

namespace DemoEDA.Infrastructure
{
    public static class StringEx
    {
        private const string MessageFormat = "{0:yyyyMMdd.HHmmss.ff} - {1} - {2} - {3}";

        public static string FormatWith(this string format, params object[] @params)
        {
            return string.Format(format, @params);
        }

        public static string FormatMessage(this string message, params object[] values)
        {
            return string.Format(
                CultureInfo.InvariantCulture,
                MessageFormat,
                DateTime.UtcNow,
                Thread.CurrentThread.GetName(),
                string.Format(CultureInfo.InvariantCulture, message, values));
        }

        private static string GetName(this Thread thread)
        {
            return !string.IsNullOrEmpty(thread.Name)
                ? thread.Name
                : thread.ManagedThreadId.ToString(CultureInfo.InvariantCulture);
        }
    }
}