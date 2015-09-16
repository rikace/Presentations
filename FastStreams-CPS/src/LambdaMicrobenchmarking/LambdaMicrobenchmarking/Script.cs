using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LambdaMicrobenchmarking
{

    public static class Script
    {
        public static Script<T> Of<T>(params Tuple<String, Func<T>>[] actions)
        {
            return Script<T>.Of(actions);
        }
        public static Script<T> Of<T>(String name, Func<T> action)
        {
            return Of(Tuple.Create(name, action));
        }
    }
    public class Script<T>
    {
        static public int Iterations { get; set; }
        static public int WarmupIterations { get; set; }

        private List<Tuple<String, Func<T>>> actions { get; set; }

        private Script(params Tuple<String, Func<T>>[] actions)
        {
            this.actions = actions.ToList();
        }

        public static Script<T> Of(params Tuple<String, Func<T>>[] actions)
        {
            return new Script<T>(actions);
        }

        public Script<T> Of(String name, Func<T> action)
        {
            actions.Add(Tuple.Create(name,action));
            return this;
        }

        public Script<T> WithHead()
        {
            Console.WriteLine("{0,-25} \t{1,10} {2,6:0.00} {3,6:0.00} {4,5}", "Benchmark", "Mean", "Mean-Error", "Sdev", "Unit");
            return this;
        }

        public Script<T> RunAll()
        {
            actions.Select(action => new Run<T>(action)).ToList().ForEach(run => run.Measure());
            return this;
        }
    }
}
