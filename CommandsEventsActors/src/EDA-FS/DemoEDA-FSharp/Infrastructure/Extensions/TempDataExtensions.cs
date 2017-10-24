using System.Web.Mvc;

namespace DemoEDA.Infrastructure
{
    public static class TempDataExtensions
    {
        public static string SuccessMessage(this TempDataDictionary tempData)
        {
            return tempData["SuccessMessage"] as string;
        }

        public static void SuccessMessage(this TempDataDictionary tempData, string message, params object[] args)
        {
            tempData["SuccessMessage"] = string.Format(message, args);
        }


        public static string ErrorMessage(this TempDataDictionary tempData)
        {
            return tempData["ErrorMessage"] as string;
        }

        public static void ErrorMessage(this TempDataDictionary tempData, string message, params object[] args)
        {
            tempData["ErrorMessage"] = string.Format(message, args);
        }
    }
}