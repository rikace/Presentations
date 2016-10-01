using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace KMeans.CSharp
{
    public class KMeansSimple
    {
        public KMeansSimple(double[][] data)
        {
            this.data = data;
            this.N = data[0].Length;
        }

        private double Dist(double[] u, double[] v)
        {
            double results = Enumerable.Range(0, u.Length)
                    .Select(i => Math.Pow(u[i] - v[i], 2.0)).Sum();
            return results;
        }

        protected int N;
        protected double[][] data;

        protected double[] getNearestCentroid(double[][] centriods, double[] u)
        {
            var res = 0;
            var minDist = Dist(u, centriods[0]);
            for (int i=1; i<centriods.Length; i++)
            {
                var dist = Dist(u, centriods[i]);
                if (dist < minDist)
                {
                    res = i;
                    minDist = dist;
                }
            }
            return centriods[res];
        }

        protected virtual double[][] updateCentroids(double[][] centroids)
        {
            var result =
              data
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
                }).ToArray();
            return Sort(result);
        }

        protected double[][] Sort(double[][] array)
        {
            Array.Sort(array, (a, b) => {
                for (var i = 0; i < N; i++)
                    if (a[i] != b[i])
                        return a[i].CompareTo(b[i]);
                return 0;
            });
            return array;
        }

        public double[][] Run(double[][] initialCentroids)
        {
            var centroids = initialCentroids;
            for (int i=0; i<=1000; i++)
            {
                var newCentroids = updateCentroids(centroids);
                var error = double.MaxValue;
                if (centroids.Length == newCentroids.Length)
                {
                    error = 0;
                    for (var j = 0; j < centroids.Length; j++)
                        error += Dist(centroids[j], newCentroids[j]);
                }
                if (error < 1e-9)
                {
                    Console.WriteLine($"Iterations {i}");
                    return newCentroids;
                }
                centroids = newCentroids;
            }
            Console.WriteLine($"Iterations 1000");
            return centroids;
        }

    }
}
