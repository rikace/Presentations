using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace KMeans.CSharp
{
    public class KMeansSimple
    {

        protected int N;
        protected double[][] data;
        public KMeansSimple(double[][] data)
        {
            this.data = data;
            this.N = data[0].Length;
        }

        private double Dist(double[] u, double[] v)
        {
            // Calculate sum of the distances between the points of each array
            // using the Euclidean Distance
            // https://en.wikipedia.org/wiki/Euclidean_distance
            double results = 0.0;
            for (var i = 0; i < u.Length; i++)
                results += Math.Pow(u[i] - v[i], 2.0);
            return results;
        }

        protected double[] getNearestCentroid(double[][] centriods, double[] u)
        {
            var res = 0;
            var minDist = Dist(u, centriods[0]);

            // Calculate the min distance between all the centroids (minBy)
            // against u
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
              // classify data by nearest centroid
              data.GroupBy(u => getNearestCentroid(centroids, u))
                // create a copy of the nearest-centroids
                // updating the values to get a step further
                .Select(elements => {
                    var res = new double[N];
                    foreach (var x in elements)
                        for (var i = 0; i < N; i++)
                            res[i] += x[i];

                    var M = elements.Count();
                    for (var i = 0; i < N; i++)
                        res[i] /= M;
                    return res;
                })
                .ToArray();
            // Sorting in place the updated centroids
            Array.Sort(result, (a, b) => {
                for (var i = 0; i < N; i++)
                    if (a[i] != b[i])
                        return a[i].CompareTo(b[i]);
                return 0;
            });
            return result;
        }

        public double[][] Run(double[][] initialCentroids)
        {
            // keep iterating of 1000 times, each iteration
            // the centroids classification is optimized
            var centroids = initialCentroids;
            for (int i=0; i<=1000; i++)
            {
                // run the updateCentroids function
                var newCentroids = updateCentroids(centroids);
                var error = double.MaxValue;

                if (centroids.Length == newCentroids.Length)
                {
                    error = 0;
                    // found the total error in the form
                    // of sum of the distances between
                    // the current centroids and the updated ones
                    for (var j = 0; j < centroids.Length; j++)
                        error += Dist(centroids[j], newCentroids[j]);
                }
                // repeat the iteration for the updated centroids
                // if the total error is bigger than 1e-9
                if (error < 1e-9)
                {
                    Console.WriteLine($"Iterations {i}");
                    return newCentroids;
                }
                centroids = newCentroids;
            }
            Console.WriteLine($"Iterations 1000");

            // return the best centroids after 1000 iterations
            return centroids;
        }

    }
}
