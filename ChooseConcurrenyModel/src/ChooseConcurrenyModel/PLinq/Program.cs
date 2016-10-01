using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Utilities;
using CommonHelpers;

namespace PLinq
{
    public class PLinqParallelism
    {

        static void Main()
        {

            MapReduceUtil.DuplicatFiles();

            Console.ReadLine();

            FromZeroToPLINQ();

            Console.ReadLine();


            var stopWatch = new Stopwatch();
            stopWatch.Start();
            NoForcedParallelism();
            stopWatch.Stop();
            Console.WriteLine("Query without forced parallelism ran in {0} ms.", stopWatch.ElapsedMilliseconds);
            stopWatch.Reset();
            stopWatch.Start();
            ForcedParallelism();
            stopWatch.Stop();
            Console.WriteLine("Query with forced parallelism ran in {0} ms.", stopWatch.ElapsedMilliseconds);
            Console.ReadLine();
        }

        public static void FromZeroToPLINQ()
        {
            var numbers = Enumerable.Range(1, 1000000).ToArray();
            var allocNums = Enumerable.Range(1, 10000000).ToArray();

            BenchPerformance.Time("LINQ", () =>
            {
                var results = (from n in numbers
                               where SimpleAlghos.isPrime(n)
                               select n).ToArray();
            }, 2);


            BenchPerformance.Time("PLINQ", () =>
            {
                var results = (from n in numbers.AsParallel()
                               where SimpleAlghos.isPrime(n)
                               select n).ToArray();
            }, 2);


            BenchPerformance.Time("PLINQ REF TYPE", () =>
            {
                var results = (from n in allocNums.AsParallel()
                               select new {Value = n}).ToArray();
            }, 2);

            BenchPerformance.Time("PLINQ VALUE TYPE", () =>
            {
                var results = (from n in allocNums.AsParallel()
                               select new Wrapper { Value = n }).ToArray();
            }, 2);

            BenchPerformance.Time("SHARED SEQ", () =>
            {
                int[] count = new int[4];
                for (int i = 0; i < 4; i++)
                    for (int j = 0; j < 100000000; j++)
                        count[i] = count[i] + j;
            }, 2);

            BenchPerformance.Time("SHARED PLINQ", () =>
            {
                int[] count = new int[4];
                ParallelEnumerable.Range(0, 4).ForAll(i =>
                {
                    for (int j = 0; j < 100000000; j++)
                        count[i] = count[i] + j;
                });
            }, 2);

            BenchPerformance.Time("BETTER SHARED PLINQ", () =>
            {
                const int PADDING = 16;
                int[] count = new int[5 * PADDING];
                ParallelEnumerable.Range(0, 4).ForAll(i =>
                {
                    int offset = (i + 1)*PADDING;
                    for (int j = 0; j < 100000000; j++)
                        count[offset] = count[offset] + j;
                });
            }, 2);

            BenchPerformance.Time("ALTERNATIVE SHARED PLINQ", () =>
            {
                int[][] count = new int[4][];
                ParallelEnumerable.Range(0, 4).ForAll(i =>
                {
                    count[i] = new int[1];
                    for (int j = 0; j < 100000000; j++)
                        count[i][0] = count[i][0] + j;
                });
            }, 2);

        }

        struct Wrapper
        {
            public int Value;
        }

        public static void ParallelAggregate()
        {
            var random = new Random();
            var numbers = ParallelEnumerable.Range(1, 1000).OrderBy(i => random.Next()).ToArray();
            var result = numbers.AsParallel().Aggregate(() => new double[2],
                (accumulator, elem) => { accumulator[0] += elem; accumulator[1]++; return accumulator; },
                (accumulator1, accumulator2) => { accumulator1[0] += accumulator2[0]; accumulator1[1] += accumulator2[1]; return accumulator1; },
                accumulator => accumulator[0] / accumulator[1]);

            Console.WriteLine("Result: {0}", result);
            Console.ReadLine();
        }


        private static void NoForcedParallelism()
        {
            Enumerable.Range(0, 1000).AsParallel()
                              .Where(x =>
                              {
                                  Thread.SpinWait(1000000);
                                  return true;
                              })
                              .Select((x, i) => i)
                              .ToArray();
        }

        private static void ForcedParallelism()
        {
            Enumerable.Range(0, 1000).AsParallel()
                              .WithExecutionMode(ParallelExecutionMode.ForceParallelism)
                              .Where(x =>
                              {
                                  Thread.SpinWait(1000000);
                                  return true;
                              })
                              .Select((x, i) => i)
                              .ToArray();
        }
    }
}