using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;

namespace Functional
{
    public static partial class Functional
    {
        public static Func<T1, Func<T2, R>> Curry<T1, T2, R>(this Func<T1, T2, R> func)
            => (T1 a) => (T2 b) => func(a, b);

        public static Func<T1, Func<T2, Func<T3, R>>> Curry<T1, T2, T3, R>(this Func<T1, T2, T3, R> func)
            => (T1 a) => (T2 b) => (T3 c) => func(a, b, c);

        public static Func<T1, Func<T2, Func<T3, Func<T4, R>>>> Curry<T1, T2, T3, T4, R>(
            this Func<T1, T2, T3, T4, R> func)
            => (T1 a) => (T2 b) => (T3 c) => (T4 d) => func(a, b, c, d);

        public static Func<T1, Func<T2, Func<T3, Func<T4, Func<T5, R>>>>> Curry<T1, T2, T3, T4, T5, R>(
            this Func<T1, T2, T3, T4, T5, R> func)
            => (T1 a) => (T2 b) => (T3 c) => (T4 d) => (T5 e) => func(a, b, c, d, e);

        public static Func<T1, Func<T2, Func<T3, Func<T4, Func<T5, Func<T6, R>>>>>> Curry<T1, T2, T3, T4, T5, T6, R>(
            this Func<T1, T2, T3, T4, T5, T6, R> func)
            => (T1 a) => (T2 b) => (T3 c) => (T4 d) => (T5 e) => (T6 f) => func(a, b, c, d, e, f);

        public static Func<T1, Func<T2, T3, R>> CurryFirst<T1, T2, T3, R>
           (this Func<T1, T2, T3, R> @this) => t1 => (t2, t3) => @this(t1, t2, t3);

        public static Func<T, T> Tap<T>(Action<T> act) => x => { act(x); return x; };



        public static Func<T1, Func<T2, Func<T3, Func<T4, Func<T5, Func<T6, Func<T7, TR>>>>>>> Curry<T1, T2, T3, T4, T5, T6, T7, TR>(this Func<T1, T2, T3, T4, T5, T6, T7, TR> func)
        => p1 => p2 => p3 => p4 => p5 => p6 => p7 => func(p1, p2, p3, p4, p5, p6, p7);


        public static Func<T1, Func<T2, Func<T3, Func<T4, Func<T5, Func<T6, Func<T7, Func<T8, TR>>>>>>>> Curry<T1, T2, T3, T4, T5, T6, T7, T8, TR>(this Func<T1, T2, T3, T4, T5, T6, T7, T8, TR> func)
        => p1 => p2 => p3 => p4 => p5 => p6 => p7 => p8 => func(p1, p2, p3, p4, p5, p6, p7, p8);


        public static Func<T1, Func<T2, Func<T3, Func<T4, Func<T5, Func<T6, Func<T7, Func<T8, Func<T9, TR>>>>>>>>> Curry<T1, T2, T3, T4, T5, T6, T7, T8, T9, TR>(this Func<T1, T2, T3, T4, T5, T6, T7, T8, T9, TR> func)
        => p1 => p2 => p3 => p4 => p5 => p6 => p7 => p8 => p9 => func(p1, p2, p3, p4, p5, p6, p7, p8, p9);


        public static Func<T1, Action<T2>> Curry<T1, T2>(this Action<T1, T2> action)
        => p1 => p2 => action(p1, p2);


        public static Func<T1, Func<T2, Action<T3>>> Curry<T1, T2, T3>(this Action<T1, T2, T3> action)
        => p1 => p2 => p3 => action(p1, p2, p3);


        public static Func<T1, Func<T2, Func<T3, Action<T4>>>> Curry<T1, T2, T3, T4>(this Action<T1, T2, T3, T4> action)
        => p1 => p2 => p3 => p4 => action(p1, p2, p3, p4);


        public static Func<T1, Func<T2, Func<T3, Func<T4, Action<T5>>>>> Curry<T1, T2, T3, T4, T5>(this Action<T1, T2, T3, T4, T5> action)
        => p1 => p2 => p3 => p4 => p5 => action(p1, p2, p3, p4, p5);


        public static Func<T1, Func<T2, Func<T3, Func<T4, Func<T5, Action<T6>>>>>> Curry<T1, T2, T3, T4, T5, T6>(this Action<T1, T2, T3, T4, T5, T6> action)
        => p1 => p2 => p3 => p4 => p5 => p6 => action(p1, p2, p3, p4, p5, p6);


        public static Func<T1, Func<T2, Func<T3, Func<T4, Func<T5, Func<T6, Action<T7>>>>>>> Curry<T1, T2, T3, T4, T5, T6, T7>(this Action<T1, T2, T3, T4, T5, T6, T7> action)
        => p1 => p2 => p3 => p4 => p5 => p6 => p7 => action(p1, p2, p3, p4, p5, p6, p7);


        public static Func<T1, Func<T2, Func<T3, Func<T4, Func<T5, Func<T6, Func<T7, Action<T8>>>>>>>> Curry<T1, T2, T3, T4, T5, T6, T7, T8>(this Action<T1, T2, T3, T4, T5, T6, T7, T8> action)
        => p1 => p2 => p3 => p4 => p5 => p6 => p7 => p8 => action(p1, p2, p3, p4, p5, p6, p7, p8);


        public static Func<T1, Func<T2, Func<T3, Func<T4, Func<T5, Func<T6, Func<T7, Func<T8, Action<T9>>>>>>>>> Curry<T1, T2, T3, T4, T5, T6, T7, T8, T9>(this Action<T1, T2, T3, T4, T5, T6, T7, T8, T9> action)
        => p1 => p2 => p3 => p4 => p5 => p6 => p7 => p8 => p9 => action(p1, p2, p3, p4, p5, p6, p7, p8, p9);


        public static Func<T2, Func<T1, R>> Swap<T1, T2, R>(Func<T1, Func<T2, R>> func)
        => p2 => p1 => func(p1)(p2);
    }
}
        
  