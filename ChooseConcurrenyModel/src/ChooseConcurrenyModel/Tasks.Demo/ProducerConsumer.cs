using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Tasks.Demo
{
    public class MultipleProducerConsumer
    {

        static void Start()
        {
            var results = new BlockingCollection<double>();
            var tasks = new List<Task>();
            var consume1 = Task.Factory.StartNew(() => DisplayResults(results));
            var consume2 = Task.Factory.StartNew(() => DisplayResults(results));
            var consume3 = Task.Factory.StartNew(() => DisplayResults(results));
            var consume4 = Task.Factory.StartNew(() => DisplayResults(results));

            for (int item = 1; item < 100; item++)
            {
                var value = item;
                var compute = Task.Factory.StartNew(() =>
                {
                    var calcResult = CalcSumRoot(value);
                    Console.Write("\nProducing item: {0}", calcResult);
                    results.TryAdd(calcResult);
                });
                tasks.Add(compute);
            }

            Task.Factory.ContinueWhenAll(tasks.ToArray(),
             result =>
             {
                 results.CompleteAdding();
                 Console.Write("\nCompleted adding.");
             });

            Console.ReadLine();

        }

        private static double CalcSumRoot(int root)
        {
            double result = 0;
            for (int i = 1; i < 10000000; i++)
            {
                result += Math.Exp(Math.Log(i) / root);
            }
            return result;
        }

        private static void DisplayResults(BlockingCollection<double> results)
        {
            foreach (var item in results.GetConsumingEnumerable())
            {
                Console.Write("\nConsuming item: {0}", item);
            }
        }
    }
}
