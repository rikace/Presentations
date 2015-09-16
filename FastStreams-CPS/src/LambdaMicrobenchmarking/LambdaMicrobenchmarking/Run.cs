using MathNet.Numerics.Distributions;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LambdaMicrobenchmarking
{
    public class Run<T>
    {
        private string title;
        private double[] measurements;
        private Func<T> func;
        private static int iterations = 10;

        public Run(Tuple<String, Func<T>> runTuple)
        {
            this.measurements = new double[iterations];
            this.title = runTuple.Item1;
            this.func = runTuple.Item2;
        }

        public int GetN()
        {
            return iterations;
        }

        public double GetMean()
        {
            if (GetN() > 0)
            {
                return GetSum() / GetN();
            }
            else
            {
                return Double.NaN;
            }
        }

        public double GetSum()
        {
            if (GetN() > 0)
            {
                double s = 0;
                for (int i = 0; i < GetN(); i++)
                {
                    s += measurements[i];
                }
                return s;
            }
            else
            {
                return Double.NaN;
            }
        }

        public double GetStandardDeviation()
        {
            return Math.Sqrt(GetVariance());
        }

        public double GetVariance()
        {
            if (GetN() > 0)
            {
                double v = 0;
                double m = GetMean();
                for (int i = 0; i < iterations; i++)
                {
                    v += Math.Pow(measurements[i] - m, 2);
                }
                return v / (GetN() - 1);
            }
            else
            {
                return Double.NaN;
            }
        }

        public double GetMeanErrorAt(double confidence)
        {
            if (GetN() <= 2) 
                return Double.NaN;
            var distribution = new StudentT(0, 1, GetN() - 1);
            double a = distribution.InverseCumulativeDistribution(1 - (1 - confidence) / 2);
            return a * GetStandardDeviation() / Math.Sqrt(GetN());
        }

        public void Measure()
        {
            //Force Full GC prior to execution
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();

            var sw = new Stopwatch();

            // First invocation to compile method.
            Compiler.ConsumeValue(func());

            for (int i = 0; i < GetN(); i++)
            {
                sw.Restart();

                Compiler.ConsumeValue(func());

                sw.Stop();

                //Force Full GC prior to execution
                GC.Collect();
                GC.WaitForPendingFinalizers();
                GC.Collect();

                TimeSpan ts = sw.Elapsed;
                measurements[i] = ts.TotalMilliseconds;
                //TODO: Introduce verbose command
                //Console.WriteLine("Milliseconds: {0}", measurements[i]);
            }

            Console.WriteLine("{0,-25}\t{1,10:0.000} {2,10:0.000} {3,6:0.000} {4,5}", 
                title, 
                GetMean(), 
                GetMeanErrorAt(0.999), 
                GetStandardDeviation(), 
                "ms/op");
        }
    }
}
