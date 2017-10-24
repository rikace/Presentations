using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;

namespace DataParallelism.CSharp
{

    public static class MapReducePLINQPartitioner
    {
        public static TResult[] MapReduce<TSource, TMapped, TKey, TResult>(
            this IList<TSource> source,
            Func<TSource, IEnumerable<TMapped>> map,
            Func<TMapped, TKey> keySelector,
            Func<IGrouping<TKey, TMapped>, TResult> reduce,
            int M, int R)
        {
            var partitioner1 = Partitioner.Create(source, true);

            var mapResults =
                partitioner1.AsParallel()
                .SelectMany(map)
                .GroupBy(keySelector)
                .ToList().AsParallel()

                .Select(reduce)
                .ToArray();
            return mapResults;
        }
    }
}
