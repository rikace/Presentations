using System;
using System.Globalization;
using System.Threading;
using DemoEDA.Infrastructure;

namespace DemoEDAFsharp.Logging
{
    internal static class ExtensionMethods
    {
        private const string MessageFormat = "{0:yyyy/MM/dd HH:mm:ss.ff} - {1} - {2}";
        private const string ExceptionFormat = "\n{0}: {1}\n{2}\n----------";

        public static string FormatMessage(this string message, params object[] values)
        {
            message = message ?? string.Empty;
            values = values ?? new object[0];

            message = FormatException(message, values);

            return string.Format(
                CultureInfo.InvariantCulture,
                MessageFormat,
                DateTime.UtcNow, // we always want the *real* point in time
                Thread.CurrentThread.GetName(),
                string.Format(CultureInfo.InvariantCulture, message, values));
        }

        private static string FormatException(string message, object[] values)
        {
            if (values.Length == 1 && values[0] is Exception)
                message = message + (values[0] as Exception).FormatException();

            return message;
        }

        private static string FormatException(this Exception exception)
        {
            if (exception == null)
                return string.Empty;

            string message = ExceptionFormat.FormatWith(exception.GetType(), exception.Message, exception.StackTrace);
            return message + exception.InnerException.FormatException();
        }

        private static string GetName(this Thread thread)
        {
            return !string.IsNullOrEmpty(thread.Name)
                ? thread.Name
                : thread.ManagedThreadId.ToString(CultureInfo.InvariantCulture);
        }
    }
}