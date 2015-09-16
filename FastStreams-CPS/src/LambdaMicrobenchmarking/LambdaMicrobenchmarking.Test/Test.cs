using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LambdaMicrobenchmarking.Test
{
    public class Test
    {
        public static void Main(string[] args)
        {
            var N = 10000000;
            var vHi = Enumerable.Range(1, 10000).Select(x => (long)x).ToArray();
            var vLow = Enumerable.Range(1, 1000).Select(x => (long)x).ToArray();
            var v = Enumerable.Range(1, N).Select(x => (long)x % 1000).ToArray();

            Func<long> sumLinq = () => v.Sum();
            Func<long> sumSqLinq = () => v.Select(x => x * x).Sum();
            Func<long> sumSqEvenLinq = () => v.Where(x => x % 2 == 0).Select(x => x * x).Sum();
            Func<long> cartLinq = () => (from x in vHi
                                         from y in vLow
                                         select x * y).Sum();

            Script
                .Of(new [] {
                    Tuple.Create("sumLinq",   sumLinq), 
                    Tuple.Create("sumSqLinq", sumSqLinq), 
                    Tuple.Create("sumSqEvensLinq", sumSqEvenLinq),
                    Tuple.Create("cartLinq", cartLinq)})
                .WithHead()
                .RunAll();

            Console.Out.WriteLine("=====================================================================");
           
            Script.Of(
                Tuple.Create("sumLinq",   sumLinq),
                Tuple.Create("sumSqLinq", sumSqLinq),
                Tuple.Create("sumSqEvensLinq", sumSqEvenLinq),
                Tuple.Create("cartLinq", cartLinq))
                .RunAll();

            Console.Out.WriteLine("=====================================================================");

            Script
                .Of("sumLinq", sumLinq)
                .Of("sumSqLinq", sumSqLinq)
                .Of("sumSqEvensLinq", sumSqEvenLinq)
                .Of("cartLinq", cartLinq)
                .RunAll();
        }
    }
}
