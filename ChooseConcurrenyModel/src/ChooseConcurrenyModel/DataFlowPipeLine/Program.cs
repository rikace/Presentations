using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Threading.Tasks.Dataflow;

namespace DataFlowPipeLine
{
    /*We start our process with BufferBlock. This block holds items to pass it to the next blocks in the flow. We restrict it to the five-items capacity, specifying the BoundedCapacity option value. This means that when there will be five items in this block, it will stop accepting new items until one of the existing items pass to the next blocks.

The next block type is TransformBlock. This block is intended for a data transformation step. Here we define two transformation blocks, one of them creates decimals from integers, and the second one creates a string from a decimal value. There is a MaxDegreeOfParallelism option for this block, specifying the maximum simultaneous worker threads.

The last block is the ActionBlock type. This block will run a specified action on every incoming item. We use this block to print our items to the console.*/
    class Program
    {
        static void Main(string[] args)
        {
            var t = ProcessAsynchronously();
            t.GetAwaiter().GetResult();
        }

        async static Task ProcessAsynchronously()
        {
            var cts = new CancellationTokenSource();

            Task.Run(() =>
            {
                if (Console.ReadKey().KeyChar == 'c')
                    cts.Cancel();
            });

            var inputBlock = new BufferBlock<int>(
                new DataflowBlockOptions { BoundedCapacity = 5, CancellationToken = cts.Token });

            var filter1Block = new TransformBlock<int, decimal>(
                n =>
                {
                    decimal result = Convert.ToDecimal(n * 0.97);
                    Console.WriteLine("Filter 1 sent {0} to the next stage on thread id {1}", result,
                        Thread.CurrentThread.ManagedThreadId);
                    Thread.Sleep(TimeSpan.FromMilliseconds(100));
                    return result;
                }
                , new ExecutionDataflowBlockOptions { MaxDegreeOfParallelism = 4, CancellationToken = cts.Token });

            var filter2Block = new TransformBlock<decimal, string>(
                n =>
                {
                    string result = string.Format("--{0}--", n);
                    Console.WriteLine("Filter 2 sent {0} to the next stage on thread id {1}", result,
                        Thread.CurrentThread.ManagedThreadId);
                    Thread.Sleep(TimeSpan.FromMilliseconds(100));
                    return result;
                }
                , new ExecutionDataflowBlockOptions { MaxDegreeOfParallelism = 4, CancellationToken = cts.Token });

            var outputBlock = new ActionBlock<string>(
                s =>
                {
                    Console.WriteLine("The final result is {0} on thread id {1}",
                        s, Thread.CurrentThread.ManagedThreadId);
                }
                , new ExecutionDataflowBlockOptions { MaxDegreeOfParallelism = 4, CancellationToken = cts.Token });


            inputBlock.LinkTo(filter1Block, new DataflowLinkOptions { PropagateCompletion = true });
            filter1Block.LinkTo(filter2Block, new DataflowLinkOptions { PropagateCompletion = true });
            filter2Block.LinkTo(outputBlock, new DataflowLinkOptions { PropagateCompletion = true });

            try
            {
                Parallel.For(0, 20, new ParallelOptions { MaxDegreeOfParallelism = 4, CancellationToken = cts.Token }
                , i =>
                {
                    Console.WriteLine("added {0} to source data on thread id {1}", i, Thread.CurrentThread.ManagedThreadId);
                    inputBlock.SendAsync(i).GetAwaiter().GetResult();
                });
                inputBlock.Complete();
                await outputBlock.Completion;
                Console.WriteLine("Press ENTER to exit.");
            }
            catch (OperationCanceledException)
            {
                Console.WriteLine("Operation has been canceled! Press ENTER to exit.");
            }

            Console.ReadLine();
        }
    }

}
