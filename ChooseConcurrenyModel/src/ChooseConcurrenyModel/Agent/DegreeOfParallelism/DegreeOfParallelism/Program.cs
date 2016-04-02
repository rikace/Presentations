using System;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks.Dataflow;

namespace DegreeOfParallelism
{
    class Program
    {
        
        static void Main(string[] args)
        {
            int processorCount = Environment.ProcessorCount;
            int messageCount = processorCount;
            TimeSpan elapsedTime;
            elapsedTime = ComputeTime(1, messageCount);
            Console.WriteLine("Degree of parallelism = {0}; message count = {1}; " +
               "elapsed time = {2}ms.", 1, messageCount, (int)elapsedTime.TotalMilliseconds);

            elapsedTime = ComputeTime(processorCount, messageCount);
            Console.WriteLine("Degree of parallelism = {0}; message count = {1}; " +
               "elapsed time = {2}ms.", processorCount, messageCount, (int)elapsedTime.TotalMilliseconds);

            Console.WriteLine("Finished. Press any key to exit.");
            Console.ReadLine();
        }

        static TimeSpan ComputeTime(int maxDegreeOfParallelism, int messageCount)
        {
            var actionBlock = new ActionBlock<int>(
               millisecondsTimeout => Thread.Sleep(millisecondsTimeout),
               new ExecutionDataflowBlockOptions
               {
                   MaxDegreeOfParallelism = maxDegreeOfParallelism
               });

            Stopwatch sw = new Stopwatch();
            sw.Start();

            for (int i = 0; i < messageCount; i++)
            {
                actionBlock.Post(1000);
            }
            actionBlock.Complete();
            actionBlock.Completion.Wait();
            sw.Stop();

            return sw.Elapsed;
        }
    }
}
