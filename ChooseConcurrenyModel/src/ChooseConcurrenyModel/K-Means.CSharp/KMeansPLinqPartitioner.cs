using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using KMeans.CSharp;

namespace KMeans.CSharp
{
    public class KMeansPLinqPartitioner : KMeansSimple
    {
        public KMeansPLinqPartitioner(double[][] data) : base(data)
        { }

        protected override double[][] updateCentroids(double[][] centroids)
        {
            // https://msdn.microsoft.com/en-us/library/dd997411%28v=vs.110%29.aspx?f=255&MSPPError=-2147217396
            var partitioner = Partitioner.Create(data, true); // load balance work
            var result =
              partitioner.AsParallel()
                .WithExecutionMode(ParallelExecutionMode.ForceParallelism)
                .GroupBy(u => getNearestCentroid(centroids, u))
                .Select(elements => {
                    var res = new double[N];
                    foreach (var x in elements)
                        for (var i = 0; i < N; i++)
                            res[i] += x[i];
                    var count = elements.Count();
                    for (var i = 0; i < N; i++)
                        res[i] /= count;
                    return res;
                })
                .ToArray();
            return Sort(result);
        }
    }
}
