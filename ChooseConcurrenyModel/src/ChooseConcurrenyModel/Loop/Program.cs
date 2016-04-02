using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Net;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace LoopDemo
{
    public class Program
    {
        static void Main()
        {
            var stopWatch = new Stopwatch();

            var random = new Random();
            var numberList = Enumerable.Range(1, 10000000).OrderBy(i => random.Next());

            stopWatch.Start();
            SequentialLoop(numberList.ToArray());
            stopWatch.Stop();
            Console.WriteLine("Time in milliseconds for sequential loop: {0}", stopWatch.ElapsedMilliseconds);

            stopWatch.Reset();
            stopWatch.Start();
            ParallelForLoop(numberList.ToArray());
            stopWatch.Stop();
            Console.WriteLine("Time in milliseconds for parallel loop: {0}", stopWatch.ElapsedMilliseconds);

            Console.Write("Complete. Press <ENTER> to exit.");
            Console.ReadKey();
        }

        private static void SequentialLoop(Int32[] array)
        {
            for (var i = 0; i < array.Length; i++)
            {
                var temp = Math.Sqrt(array[i]);
            }
        }


        private static void ParallelForLoop(Int32[] array)
        {
            Parallel.For(0, array.Length, i =>
            {
                var temp = Math.Sqrt(array[i]);
            }
            );
        }

        public static void StopLoop()
        {
            char[] delimiters = { ' ', ',', '.', ';', ':', '-', '_', '/', '"', '(', ')', '\u000A' };
            var client = new WebClient();
            const string headerText = "Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.1; Trident/6.0)";
            client.Headers.Add("user-agent", headerText);
            var words = client.DownloadString(@"http://www.gutenberg.org/files/2009/2009.txt");
            var wordList = words.Split(delimiters, StringSplitOptions.RemoveEmptyEntries).Where(word => word.Length > 5).ToList();
            wordList.Sort();
            var loopResult = Parallel.ForEach(wordList, (currentWord, loopState) =>
            {
                if (!currentWord.Equals("immutability"))
                    Console.WriteLine(currentWord);
                else
                {
                    loopState.Stop();
                    Console.WriteLine(currentWord);
                    Console.WriteLine("Loop stopped: {0}", loopState.IsStopped);
                }
            });
            Console.WriteLine("Loop Completed : {0}", loopResult.IsCompleted);
            Console.ReadLine();
        }

        public static void BreakALoop()
        {
            char[] delimiters = { ' ', ',', '.', ';', ':', '-', '_', '/', '\u000A' };
            var client = new WebClient();
            const string headerText = "Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.1; Trident/6.0)";
            client.Headers.Add("user-agent", headerText);
            var words = client.DownloadString(@"http://www.gutenberg.org/files/2009/2009.txt");
            var wordList = words.Split(delimiters, StringSplitOptions.RemoveEmptyEntries).ToList();
            var loopResult = Parallel.ForEach(wordList, (parm, loopState) =>
            {
                if (parm.Equals("immutability"))
                {
                    Console.WriteLine(parm);
                    loopState.Break();
                }
            });
            Console.WriteLine("Loop LowestBreak Iteration : {0}", loopResult.LowestBreakIteration);
            Console.WriteLine("Loop Completed : {0}", loopResult.IsCompleted);
            Console.ReadLine();
        }


        public static void PartitionData()
        {
            var stopWatch = new Stopwatch();

            var random = new Random();
            var numbers = Enumerable.Range(1, 10000000).OrderBy(i => random.Next()).ToArray();

            stopWatch.Start();
            NoPartitioning(numbers);
            stopWatch.Stop();
            Console.WriteLine("Time in milliseconds for no partitioning: {0}", stopWatch.ElapsedMilliseconds);

            stopWatch.Reset();
            stopWatch.Start();
            DefaultPartitioning(numbers);
            stopWatch.Stop();
            Console.WriteLine("Time in milliseconds for default partitioning: {0}", stopWatch.ElapsedMilliseconds);

            stopWatch.Reset();
            stopWatch.Start();
            CustomPartitioning(numbers);
            stopWatch.Stop();
            Console.WriteLine("Time in milliseconds for custom partitioning: {0}", stopWatch.ElapsedMilliseconds);

            Console.Write("Complete. Press <ENTER> to exit.");
            Console.ReadKey();
        }

        private static void NoPartitioning(Int32[] numbers)
        {
            Parallel.ForEach(numbers, currentNumber =>
            {
                var temp = Math.Sqrt(currentNumber);
            });
        }

        private static void DefaultPartitioning(Int32[] numbers)
        {
            var partitioner = Partitioner.Create(numbers);
            Parallel.ForEach(partitioner, currentNumber =>
            {
                var temp = Math.Sqrt(currentNumber);
            });
        }

        private static void CustomPartitioning(Int32[] numbers)
        {
            var partitioner = Partitioner.Create(0, numbers.Count(), 100000);
            Parallel.ForEach(partitioner, range =>
            {
                for (var index = range.Item1; index < range.Item2; index++)
                {
                    var temp = Math.Sqrt(numbers[index]);
                }
            });
        }


        public static void DegreeOfParallelism()
        {
            var stopWatch = new Stopwatch();

            var random = new Random();
            var numberList = Enumerable.Range(1, 1000000).OrderBy(i => random.Next());

            stopWatch.Start();
            DefaultParallelism(numberList.ToArray());
            stopWatch.Stop();
            Console.WriteLine("Time in milliseconds for default parallelism: {0}", stopWatch.ElapsedMilliseconds);

            stopWatch.Reset();
            stopWatch.Start();
            LimitedParallelism(numberList.ToArray());
            stopWatch.Stop();
            Console.WriteLine("Time in milliseconds for MaxDegreeOfParallelism: {0}", stopWatch.ElapsedMilliseconds);

            Console.Write("Complete. Press <ENTER> to exit.");
            Console.ReadKey();
        }

        private static void DefaultParallelism(Int32[] array)
        {
            Parallel.For(0, array.Length, i =>
            {
                var temp = Math.Sqrt(array[i]);
            }
            );
        }


        private static void LimitedParallelism(Int32[] array)
        {
            var options = new ParallelOptions()
            {
                MaxDegreeOfParallelism = 4
            };

            Parallel.For(0, array.Length, options, i =>
            {
                var temp = Math.Sqrt(array[i]);
            });
        }
        public static void ThreadLocalStorage()
        {
            char[] delimiters = { ' ', ',', '.', ';', ':', '-', '_', '/', '\u000A' };
            var client = new WebClient();
            const string headerText = "Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.1; Trident/6.0)";
            client.Headers.Add("user-agent", headerText);
            var words = client.DownloadString(@"http://www.gutenberg.org/files/2009/2009.txt");
            var wordList = words.Split(delimiters, StringSplitOptions.RemoveEmptyEntries).ToList();

            //word count total
            Int32 total = 0;

            // Sting is type of source elements
            // int32 is type of thread-local count variable
            // wordlist is the source collection
            // ()=>0 initializes local variable
            Parallel.ForEach<String, Int32>(wordList, () => 0,
                (word, loopstate, count) =>  // method invoked on each iteration of loop
                {
                    if (word.Equals("species"))
                    {
                        count++; // increment the count
                    }
                    return count;
                }, (result) => Interlocked.Add(ref total, result)); // executed when all loops have completed

            Console.WriteLine("The word specied occured {0} times.", total);
            Console.ReadLine();
        }
    }
}