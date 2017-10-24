namespace DemoEDAFsharp.Logging
{
    public interface ILog
    {
        void Verbose(string message, params object[] values);
        void Debug(string message, params object[] values);
        void Info(string message, params object[] values);
        void Warn(string message, params object[] values);
        void Error(string message, params object[] values);
        void Fatal(string message, params object[] values);
    }
}