using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace Tasks.Demo
{
    public class Continuation
    {
        public static void ContinueToDispose()
        {
            try
            {
                var client = new WebClient();
                const string headerText = "Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.1; Trident/6.0)";
                client.Headers.Add("user-agent", headerText);

                Task.Factory.StartNew(() =>
                {
                    Console.WriteLine("Antecedent running.");
                    char[] delimiters = { ' ', ',', '.', ';', ':', '-', '_', '/', '\u000A' };
                    var words = client.DownloadString(@"http://www.gutenberg.org/files/2009/2009.txt");
                    var wordArray = words.Split(delimiters, StringSplitOptions.RemoveEmptyEntries);
                    Console.WriteLine("Word count for Origin of Species: {0}", wordArray.Count());
                }
                ).ContinueWith(antecedent =>
                {
                    Console.WriteLine("Continuation running");
                    client.Dispose();
                }).Wait();

                Console.WriteLine("Complete. Please hit <Enter> to exit.");
                Console.ReadLine();
            }
            catch (AggregateException aEx)
            {
                foreach (var ex in aEx.InnerExceptions)
                {
                    Console.WriteLine("An exception has occured: {0}" + ex.Message);
                }
            }
        }

        // Group count wodrs
        public static void CountWordsContinue()
        {
            char[] delimiters = { ' ', ',', '.', ';', ':', '-', '_', '/', '\u000A' };

            try
            {

                Task<string[]> task1 = Task.Factory.StartNew(() =>
                {
                    var client = new WebClient();
                    const string headerText = "Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.1; Trident/6.0)";
                    client.Headers.Add("user-agent", headerText);
                    var words = client.DownloadString(@"http://www.gutenberg.org/files/2009/2009.txt");
                    var wordArray = words.Split(delimiters, StringSplitOptions.RemoveEmptyEntries);
                    Console.WriteLine("Word count for Origin of Species: {0}", wordArray.Count());
                    Console.WriteLine();
                    return wordArray;
                }
                );

                task1.ContinueWith(antecedent =>
                {
                    var wordsByUsage =
                        antecedent.Result.Where(word => word.Length > 5)
                                  .GroupBy(word => word)
                                  .OrderByDescending(grouping => grouping.Count())
                                  .Select(grouping => grouping.Key);
                    var commonWords = (wordsByUsage.Take(5)).ToArray();
                    Console.WriteLine("The 5 most commonly used words in Origin of Species:");
                    Console.WriteLine("----------------------------------------------------");
                    foreach (var word in commonWords)
                    {
                        Console.WriteLine(word);
                    }
                }).Wait();

                Console.WriteLine();
                Console.WriteLine("Complete. Please hit <Enter> to exit.");
                Console.ReadLine();
            }
            catch (AggregateException aEx)
            {
                foreach (var ex in aEx.InnerExceptions)
                {
                    Console.WriteLine("An exception has occured: {0}" + ex.Message);
                }
            }
        }

        public static void CountWordsMultipleTasks()
        {
            char[] delimiters = { ' ', ',', '.', ';', ':', '-', '_', '/', '\u000A' };
            const string headerText = "Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.1; Trident/6.0)";
            var dictionary = new Dictionary<string, string>
                {
                    {"Origin of Species", "http://www.gutenberg.org/files/2009/2009.txt"},
                    {"Beowulf", "http://www.gutenberg.org/files/16328/16328-8.txt"},
                    {"Ulysses", "http://www.gutenberg.org/files/4300/4300.txt"}
                };
            try
            {
                var tasks = new List<Task<KeyValuePair<string, int>>>();

                foreach (var pair in dictionary)
                {
                    tasks.Add(Task.Factory.StartNew(stateObj =>
                    {
                        var taskData = (KeyValuePair<string, string>)stateObj;
                        Console.WriteLine("Starting task for {0}", taskData.Key);
                        var client = new WebClient();
                        client.Headers.Add("user-agent", headerText);
                        var words = client.DownloadString(taskData.Value);
                        var wordArray = words.Split(delimiters, StringSplitOptions.RemoveEmptyEntries);
                        return new KeyValuePair<string, int>(taskData.Key, wordArray.Count());
                    }, pair));
                }
                Task.Factory.ContinueWhenAll(tasks.ToArray(), antecedents =>
                {
                    foreach (var antecedent in antecedents)
                    {
                        Console.WriteLine("Book Title: {0}", antecedent.Result.Key);
                        Console.WriteLine("Word count: {0}", antecedent.Result.Value);
                    }

                }).Wait();
            }
            catch (AggregateException aEx)
            {
                foreach (var ex in aEx.InnerExceptions)
                {
                    Console.WriteLine("An exception has occured: {0}" + ex.Message);
                }
            }
            Console.WriteLine("Complete. Press <Enter> to exit.");
            Console.ReadLine();
        }


        public static void TaskContinueOptions()
        {
            var tokenSource1 = new CancellationTokenSource();
            var token1 = tokenSource1.Token;

            var tokenSource2 = new CancellationTokenSource();
            var token2 = tokenSource2.Token;

            try
            {
                var task1 = Task.Factory.StartNew(() =>
                {
                    Console.WriteLine("Task #1 is running.");
                    //wait a bit
                    Thread.Sleep(2000);
                }, token1);

                task1.ContinueWith(antecedent => Console.WriteLine("Task #1 completion continuation."), TaskContinuationOptions.OnlyOnRanToCompletion);

                task1.ContinueWith(antecedent => Console.WriteLine("Task #1 cancellation continuation."), TaskContinuationOptions.OnlyOnCanceled);

                var task2 = Task.Factory.StartNew(() =>
                {
                    Console.WriteLine("Task #2 is running.");
                    //wait a bit
                    Thread.Sleep(2000);
                }, token2);

                task2.ContinueWith(antecedent => Console.WriteLine("Task #2 completion continuation."), TaskContinuationOptions.OnlyOnRanToCompletion);

                task2.ContinueWith(antecedent => Console.WriteLine("Task #2 cancellation continuation."), TaskContinuationOptions.OnlyOnCanceled);
            }
            catch (AggregateException aEx)
            {
                foreach (var ex in aEx.InnerExceptions)
                {
                    Console.WriteLine("Caught exception: {0}", ex.Message);
                }
            }
            tokenSource2.Cancel();
            Console.ReadLine();
        }

        public static void PipelineTasks()
        {

            try
            {
                var producer = Task.Factory.StartNew(() =>
                {
                    char[] delimiters = { ' ', ',', '.', ';', ':', '-', '_', '/', '\u000A' };
                    var client = new WebClient();
                    const string headerText = "Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.1; Trident/6.0)";
                    client.Headers.Add("user-agent", headerText);
                    try
                    {
                        var words = client.DownloadString(@"http://www.gutenberg.org/files/2009/2009.txt");
                        var wordArray = words.Split(delimiters, StringSplitOptions.RemoveEmptyEntries);
                        Console.WriteLine("Word count for Origin of Species: {0}", wordArray.Count());
                        Console.WriteLine();
                        return wordArray;
                    }
                    finally
                    {
                        client.Dispose();
                    }
                });
                Task<string[]> consumer1 = producer.ContinueWith(antecedent =>
                {
                    var wordsByUsage =
                      antecedent.Result.Where(word => word.Length > 5)
                                .GroupBy(word => word)
                                .OrderByDescending(grouping => grouping.Count())
                                .Select(grouping => grouping.Key);
                    var commonWords = (wordsByUsage.Take(5)).ToArray();
                    Console.WriteLine("The 5 most commonly used words in Origin of Species:");
                    Console.WriteLine("----------------------------------------------------");
                    foreach (var word in commonWords)
                    {
                        Console.WriteLine(word);
                    }
                    Console.WriteLine();
                    return antecedent.Result;
                }, TaskContinuationOptions.OnlyOnRanToCompletion);

                Task consumer2 = consumer1.ContinueWith(antecedent =>
                {
                    var longestWord = (antecedent.Result.OrderByDescending(w => w.Length)).First();
                    Console.WriteLine("The longest word is: {0}", longestWord);
                }, TaskContinuationOptions.OnlyOnRanToCompletion);
                consumer2.Wait();
            }
            catch (AggregateException aEx)
            {
                foreach (var ex in aEx.InnerExceptions)
                {
                    Console.WriteLine("An exception has occured: {0}", ex.Message);
                }
            }
            finally
            {
                Console.WriteLine();
                Console.WriteLine("Complete. Please hit <Enter> to exit.");
                Console.ReadLine();
            }
        }
    }
}