using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace PLinq
{
    public class PLinqParallelism
    {

        static void Main()
        {
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