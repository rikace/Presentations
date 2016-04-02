using System;
using System.Linq;
using System.Net;
using System.Threading;
using System.Threading.Tasks;

namespace Tasks.Demo
{
    class Cancel
    {
        static void Start()
        {
            //Create a cancellation token source
            var tokenSource = new CancellationTokenSource();
            //get the cancellation token
            CancellationToken token = tokenSource.Token;

            char[] delimiters = { ' ', ',', '.', ';', ':', '-', '_', '/', '\u000A' };
            const string headerText = "Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.1; Trident/6.0)";

            Task<int> task1 = Task.Factory.StartNew(() =>
            {
                // wait for the cancellation to happen
                Thread.Sleep(2000);
                var client = new WebClient();
                client.Headers.Add("user-agent", headerText);
                if (token.IsCancellationRequested)
                {
                    client.Dispose();
                    throw new OperationCanceledException(token);
                }
                else
                {
                    var words = client.DownloadString(@"http://www.gutenberg.org/files/2009/2009.txt");
                    Console.WriteLine("Starting the task for Origin of Species.");
                    var wordArray = words.Split(delimiters, StringSplitOptions.RemoveEmptyEntries);
                    return wordArray.Count();
                }
            }, token);

            Task<int> task2 = Task.Factory.StartNew(() =>
            {
                // wait for the cancellation to happen
                Thread.Sleep(2000);
                var client = new WebClient();
                client.Headers.Add("user-agent", headerText);
                if (token.IsCancellationRequested)
                {
                    client.Dispose();
                    throw new OperationCanceledException(token);
                }
                else
                {
                    var words = client.DownloadString(@"http://www.gutenberg.org/files/16328/16328-8.txt");
                    Console.WriteLine("Starting the task for Beowulf.");
                    var wordArray = words.Split(delimiters, StringSplitOptions.RemoveEmptyEntries);
                    return wordArray.Count();
                };
            }, token);

            Task<int> task3 = Task.Factory.StartNew(() =>
            {
                // wait for the cancellation to happen
                Thread.Sleep(2000);
                var client = new WebClient();
                client.Headers.Add("user-agent", headerText);
                if (token.IsCancellationRequested)
                {
                    client.Dispose();
                    throw new OperationCanceledException(token);
                }
                else
                {
                    var words = client.DownloadString(@"http://www.gutenberg.org/files/4300/4300.txt");
                    Console.WriteLine("Starting the task for Ulysses.");
                    var wordArray = words.Split(delimiters, StringSplitOptions.RemoveEmptyEntries);
                    return wordArray.Count();
                };
            }, token);


            //Cancel the token source
            tokenSource.Cancel();

            try
            {
                if (!task1.IsFaulted || !task1.IsCanceled)
                {
                    Console.WriteLine("Origin of Specied word count: {0}", task1.Result);
                }
            }
            catch (AggregateException outerEx1)
            {
                DisplayException(task1, outerEx1, "Origin of Species");
            }

            try
            {
                if (!task2.IsFaulted || !task2.IsCanceled)
                {
                    Console.WriteLine("Beowulf word count: {0}", task2.Result);
                }
            }
            catch (AggregateException outerEx2)
            {
                DisplayException(task2, outerEx2, "Beowulf");
            }

            try
            {
                if (!task3.IsFaulted || !task3.IsCanceled)
                {
                    Console.WriteLine("Ulysses word count: {0}", task3.Result);
                }
            }
            catch (AggregateException outerEx3)
            {
                DisplayException(task3, outerEx3, "Ulysses");
            }

            Console.WriteLine("Press <Enter> to exit.");
            Console.ReadLine();
        }

        private static void DisplayException(Task task, AggregateException outerEx, string bookName)
        {
            foreach (Exception innerEx in outerEx.InnerExceptions)
            {
                Console.WriteLine("Handled exception for {0}:{1}", bookName, innerEx.Message);
            }
            Console.WriteLine("Cancellation status for book {0}: {1}", bookName, task.IsCanceled);
        }

    }
}
