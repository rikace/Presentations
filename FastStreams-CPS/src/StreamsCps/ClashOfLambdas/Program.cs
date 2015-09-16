using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Linq.Expressions;
using System.Text;
using Nessos.LinqOptimizer.Base;
using Nessos.LinqOptimizer.CSharp;
using Nessos.Streams.CSharp;
using LambdaMicrobenchmarking;

namespace benchmarks
{
    public class Ref
    {
        public Ref(int num)
        {
            this.Num = num;
        }
        public int Num;
    }

    class ClashOfLambdas
    {

        static void Main(string[] args)
        {
            //////////////////////////
            // Input initialization //
            //////////////////////////
            var N = 10000000;
            var v = Enumerable.Range(1, N).Select(x => (long)x % 1000).ToArray();
            var vHi = Enumerable.Range(1, 1000000).Select(x => (long)x).ToArray();
            var vLow = Enumerable.Range(1, 10).Select(x => (long)x).ToArray();
            var refs = Enumerable.Range(1, N).Select(num => new Ref(num)).ToArray();

            ///////////////////////////
            // Benchmarks definition //
            ///////////////////////////
            Func<long> sumBaseline = () =>
            {
                var acc = 0L;
                for (int i = 0; i < v.Length; i++)
                    acc += v[i];
                return acc;
            };
            Func<long> sumOfSquaresBaseline = () =>
            {
                var acc = 0L;
                for (int i = 0; i < v.Length; i++)
                    acc += v[i] * v[i];
                return acc;
            };
            Func<long> sumOfSquaresEvenBaseline = () =>
            {
                var acc = 0L;
                for (int i = 0; i < v.Length; i++)
                    if (v[i] % 2 == 0)
                        acc += v[i] * v[i];
                return acc;
            };
            Func<long> cartBaseline = () =>
            {
                var acc = 0L;
                for (int d = 0; d < vHi.Length; d++)
                    for (int dp = 0; dp < vLow.Length; dp++)
                        acc += vHi[d] * vLow[dp];
                return acc;
            };
            Func<int> refBaseline = () =>
            {
                var count = 0;
                for (int i = 0; i < refs.Length; i++)
                    if (refs[i].Num % 5 == 0 && refs[i].Num % 7 == 0)
                        count++;
                return count;
            };
            Func<long> sumLinq = () => v.Sum();
            Func<long> sumLinqOpt = v.AsQueryExpr().Sum().Compile();
            Func<long> sumSqLinq = () => v.Select(x => x * x).Sum();
            Func<long> sumSqLinqOpt = v.AsQueryExpr().Select(x => x * x).Sum().Compile();
            Func<long> sumSqEvensLinq = () => v.Where(x => x % 2 == 0).Select(x => x * x).Sum();
            Func<long> sumSqEvenLinqOpt = v.AsQueryExpr().Where(x => x % 2 == 0).Select(x => x * x).Sum().Compile();
            Func<long> cartLinq = () => (from x in vHi
                                         from y in vLow
                                         select x * y).Sum();
            Func<long> cartLinqOpt = (from x in vHi.AsQueryExpr()
                                      from y in vLow
                                      select x * y).Sum().Compile();
            Func<long> parSumLinq = () => v.AsParallel().Sum();
            Func<long> parSumLinqOpt = v.AsParallelQueryExpr().Sum().Compile();
            Func<long> parSumSqLinq = () => v.AsParallel().Select(x => x * x).Sum();
            Func<long> parSumSqLinqOpt = v.AsParallelQueryExpr().Select(x => x * x).Sum().Compile();
            Func<long> parSumSqEvensLinq = () => v.AsParallel().Where(x => x % 2 == 0).Select(x => x * x).Sum();
            Func<long> parSumSqEvenLinqOpt = v.AsParallelQueryExpr().Where(x => x % 2 == 0).Select(x => x * x).Sum().Compile();
            Func<long> parCartLinq = () => (from x in vHi.AsParallel()
                                            from y in vLow
                                            select x * y).Sum();
            Func<long> parCartLinqOpt = (from x in vHi.AsParallelQueryExpr()
                                         from y in vLow
                                         select x * y).Sum().Compile();


            Func<int> refLinq = () => refs.Where(box => box.Num % 5 == 0).Where(box => box.Num % 7 == 0).Count();
            Func<int> refLinqOpt = refs.AsQueryExpr().Where(box => box.Num % 5 == 0).Where(box => box.Num % 7 == 0).Count().Compile();
            Func<int> parRefLinq = () => refs.AsParallel().Where(box => box.Num % 5 == 0).Where(box => box.Num % 7 == 0).Count();
            Func<int> parRefLinqOpt = refs.AsParallelQueryExpr().Where(box => box.Num % 5 == 0).Where(box => box.Num % 7 == 0).Count().Compile();

            Func<long> sumLinqStreams = () => v.AsStream().Sum();
            Func<long> sumSqStreams = () => v.AsStream().Select(x => x * x).Sum();
            Func<long> sumSqEvensStreams = () => v.AsStream().Where(x => x % 2 == 0).Select(x => x * x).Sum();
            Func<long> cartStreams = () => vHi.AsStream().SelectMany(hi => vLow.AsStream().Select(lo => lo * hi)).Sum();
            Func<long> refStreams = () => refs.AsStream().Where(box => box.Num % 5 == 0).Where(box => box.Num % 7 == 0).Count();


            //////////////////////////
            // Benchmarks execution //
            //////////////////////////
            Script<long>.Of(new Tuple<String, Func<long>>[] {
                Tuple.Create("sumBaseline", sumBaseline),
                Tuple.Create("sumSeq", sumLinq),
                Tuple.Create("sumSeqOpt", sumLinqOpt),
                Tuple.Create("sumPar", parSumLinq),
                Tuple.Create("sumParOpt", parSumLinqOpt),
                Tuple.Create("sumOfSquaresBaseline", sumOfSquaresBaseline),
                Tuple.Create("sumOfSquaresSeq", sumSqLinq),
                Tuple.Create("sumOfSquaresSeqOpt", sumSqLinqOpt),
                Tuple.Create("sumOfSquaresPar", parSumSqLinq),
                Tuple.Create("sumOfSquaresParOpt", parSumSqLinqOpt),
                Tuple.Create("sumOfSquaresEvenBaseline",sumOfSquaresEvenBaseline),
                Tuple.Create("sumOfSquaresEvenSeq", sumSqEvensLinq),
                Tuple.Create("sumOfSquaresEvenSeqOpt", sumSqEvenLinqOpt),
                Tuple.Create("sumOfSquaresEvenPar",  parSumSqEvensLinq),
                Tuple.Create("sumOfSquaresEvenParOpt",  parSumSqEvenLinqOpt),
                Tuple.Create("cartBaseline", cartBaseline),
        Tuple.Create("cartSeq", cartLinq),
                Tuple.Create("cartSeqOpt",cartLinqOpt),
                Tuple.Create("cartPar", parCartLinq),
        Tuple.Create("cartParOpt", parCartLinqOpt)})
          .WithHead()
          .RunAll();

            Script<int>.Of(new Tuple<String, Func<int>>[] {
        Tuple.Create("refBaseline", refBaseline),
        Tuple.Create("refSeq", refLinq),
        Tuple.Create("refSeqOpt",refLinqOpt),
        Tuple.Create("refPar", parRefLinq),
        Tuple.Create("refParOpt", parRefLinqOpt)})
              .RunAll();

            Script<long>.Of(new Tuple<String, Func<long>>[] {
        Tuple.Create("sumLinqStreams", sumLinqStreams),
        Tuple.Create("sumSqStreams", sumSqStreams),
        Tuple.Create("sumSqEvensStreams",sumSqEvensStreams),
        Tuple.Create("cartStreams", cartStreams),
        Tuple.Create("refStreams", refStreams)})
      .RunAll();
        }
    }
}

