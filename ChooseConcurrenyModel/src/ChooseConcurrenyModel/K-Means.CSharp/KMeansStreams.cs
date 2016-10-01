using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nessos.Streams;
using Nessos.Streams.CSharp;


namespace KMeans.CSharp
{
    public class KMeansStreams : KMeansSimple
    {
        public KMeansStreams(double[][] data) : base(data)
        {
        }

        protected override double[][] updateCentroids(double[][] centroids)
        {
            var result =
              data.AsStream()
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
