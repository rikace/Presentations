using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using FList = FCSlib.Data.Collections.List<System.Func<int>>;

namespace CSMisc
{
    public class ImmutabilityClosure
    {
        public static void LoopFunc()
        {
            List<Func<int>> actions = new List<Func<int>>();

            for (int i = 0; i < 5; i++)
                actions.Add(() => i * 2);

            foreach (var action in actions)
                Console.WriteLine(action());
        }
        public static void LoopImmutableCollections()
        {
            var ints = Enumerable.Range(1, 4);
            var actions = Enumerable.Aggregate(ints, ImmutableList.Create<Func<int>>(()=>0),
                        (f, i) => f.Add(() => i * 2));

            foreach (var action in actions)
                Console.WriteLine(action());
        }

        public static void LoopImmutable()
        {
            var ints = Enumerable.Range(1, 4);
            var actions = Enumerable.Aggregate(ints, new FList(() => 0),
                        (f, i) => FList.Cons(() => i * 2, f));
            
            foreach (var action in actions)
                Console.WriteLine(action());
        }

        public static void Loop()
        {
            List<int> ints = new List<int>();

            for (int i = 0; i < 5; i++)
                ints.Add(i * 2);

            foreach (var n in ints)
                Console.WriteLine(n);
        }
    }
}

