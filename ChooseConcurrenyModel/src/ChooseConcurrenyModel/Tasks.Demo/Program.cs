using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Text;
using System.Threading.Tasks;

namespace Tasks.Demo
{
    class Program
    {
        static void Main(string[] args)
        {

            Task<int> task1 = Task.Factory.StartNew(() =>
            {
                Console.WriteLine("Starting the task.");
                const string headerText = "Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.1; Trident/6.0)";
                var client = new WebClient();
                client.Headers.Add("user-agent", headerText);
                string words = client.DownloadString(@"http://www.gutenberg.org/files/2009/2009.txt");
                var ex = new WebException("Unable to download book contents");
                throw ex;
                return 0;
            }
            );
            try
            {
                task1.Wait();
                if (!task1.IsFaulted)
                {
                    Console.WriteLine("Task complete. Origin of Species word count: {0}", task1.Result);
                }
            }
            catch (AggregateException aggEx)
            {
                aggEx.Handle(HandleWebExceptions);
            }
            Console.WriteLine("Press <Enter> to exit.");
            Console.ReadLine();
        }

        private static bool HandleWebExceptions(Exception ex)
        {
            if (ex is WebException)
            {
                Console.WriteLine("Caught WebException: {0}", ex.Message);
                return true;
            }
            Console.WriteLine("Caught exception: {0}", ex.Message);
            return false;
        }
    }
}