using NLog;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OeipCommon
{
    public enum OeipLogLevel
    {
        OEIP_INFO,
        OEIP_WARN,
        OEIP_ERROR,
        OEIP_ALORT,
    }

    public static class LogHelper
    {
        private static ILogger Log = LogManager.GetCurrentClassLogger();
        private static void LogMessage(string message, LogLevel level)
        {
            Console.ForegroundColor = ConsoleColor.White;
            if (level > LogLevel.Info && level <= LogLevel.Warn)
                Console.ForegroundColor = ConsoleColor.Yellow;
            else if (level > LogLevel.Warn)
                Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine("{0} {1} {2}", DateTime.Now.ToLongTimeString(), level, message);
            Log.Log(level, message);
        }

        public static void LogMessage(string message)
        {
            LogMessage(message, LogLevel.Info);
        }

        public static void LogMessage(string message, OeipLogLevel logLevel)
        {
            LogLevel level = LogLevel.Info;
            switch (logLevel)
            {
                case OeipLogLevel.OEIP_INFO:
                    level = LogLevel.Info;
                    break;
                case OeipLogLevel.OEIP_WARN:
                    level = LogLevel.Warn;
                    break;
                case OeipLogLevel.OEIP_ERROR:
                    level = LogLevel.Error;
                    break;
                case OeipLogLevel.OEIP_ALORT:
                    level = LogLevel.Fatal;
                    break;
            }
            LogMessage(message, level);
        }

        public static void LogMessageEx(string message, Exception exception)
        {
            LogMessage(message + " " + exception.Message, LogLevel.Error);
            Log.Error(exception);
        }
    }
}
