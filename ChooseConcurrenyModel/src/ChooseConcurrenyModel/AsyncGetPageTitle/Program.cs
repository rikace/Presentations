using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Text;
using System.Threading.Tasks;


namespace AsyncGetPageTitle
{
    class Program
    {
        static void Main(string[] args)
        {
            var t = GetTitleCsAsync(
                "http://concurrencyfreaks.blogspot.com");
            Console.WriteLine("Returned");
            string title = t.Result;
            Console.WriteLine(title);
        }

        static async Task<string> GetTitleCsAsync(string url)
        {
            using (var w = new WebClient())
            {
                string content = await w.DownloadStringTaskAsync(url);
                return ExtractTitle(content);
            }
        }

        static Task<string> GetTitleTplAsync(string url)
        {
            var w = new WebClient();
            Task<string> contentTask = w.DownloadStringTaskAsync(url);
            return contentTask.ContinueWith(t =>
            {
                string result = ExtractTitle(t.Result);
                w.Dispose();
                return result;
            });
        }
        static string GetTitle(string url)
        {
            using (var w = new WebClient())
            {
                string content = w.DownloadString(url);
                return ExtractTitle(content);
            }
        }

        private static string ExtractTitle(string content)
        {
            const string TitleTag = "<title>";
            var comp = StringComparison.InvariantCultureIgnoreCase;
            int titleStart = content.IndexOf(TitleTag, comp);
            if (titleStart < 0)
            {
                return null;
            }
            int titleBodyStart = titleStart + TitleTag.Length;
            int titleBodyEnd = content.IndexOf("</title>", titleBodyStart, comp);
            return content.Substring(titleBodyStart, titleBodyEnd - titleBodyStart);
        }
    }

}
