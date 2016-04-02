using System;
using System.Linq;
using System.Net;
using System.Threading.Tasks;
namespace Tasks.Demo
{
    public class WordCount
    {

        int wordCount = 0;


        static void Start()
        {
           char[] delimiters = { ' ', ',', '.', ';', ':', '-', '_', '/', '\u000A' };
            const string headerText = "Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.1; Trident/6.0)";

            Task<int> task1 = Task.Factory.StartNew(() =>
            {
                Console.WriteLine("Starting first task.");
                var client = new WebClient();
                client.Headers.Add("user-agent", headerText);
                var words = client.DownloadString(@"http://www.gutenberg.org/files/2009/2009.txt");
                var wordArray = words.Split(delimiters, StringSplitOptions.RemoveEmptyEntries);
                return wordArray.Count();
            }
            );

            Task<int> task2 = Task.Factory.StartNew(() =>
            {
                Console.WriteLine("Starting second task.");
                var client = new WebClient();
                client.Headers.Add("user-agent", headerText);
                var words = client.DownloadString(@"http://www.gutenberg.org/files/16328/16328-8.txt");
                var wordArray = words.Split(delimiters, StringSplitOptions.RemoveEmptyEntries);
                return wordArray.Count();
            }
            );

            Task<int> task3 = Task.Factory.StartNew(() =>
            {
                Console.WriteLine("Starting third task.");
                var client = new WebClient();
                client.Headers.Add("user-agent", headerText);
                var words = client.DownloadString(@"http://www.gutenberg.org/files/4300/4300.txt");
                var wordArray = words.Split(delimiters, StringSplitOptions.RemoveEmptyEntries);
                return wordArray.Count();
            }
            );

            Console.WriteLine("task1 is complete. Origin of Species word count: {0}", task1.Result);
            Console.WriteLine("task2 is complete. Beowulf word count: {0}", task2.Result);
            Console.WriteLine("task3 is complete. Ulysses word count: {0}", task3.Result);
            Console.WriteLine("Press <Enter> to exit.");
            Console.ReadLine();
        }
    }
}
