using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Functional
{
    public static partial class Functional
    {
        public static Func<T1, TR> Compose<T1, T2, TR>(this Func<T1, T2> f1, Func<T2, TR> f2) => v => f2(f1(v));

        public static Func<T1, T2, TR> Compose<T1, T2, T3, TR>(this Func<T1, T2, T3> f1, Func<T3, TR> f2)
            => (a, b) => f2(f1(a, b));

        public static Func<T1, T2, T3, TR> Compose<T1, T2, T3, T4, TR>(this Func<T1, T2, T3, T4> f1, Func<T4, TR> f2)
            => (a, b, c) => f2(f1(a, b, c));

        public static Func<T1, TR> Compose<T1, T2, T3, TR>(
            Func<T1, T2> f1, Func<T2, T3> f2, Func<T3, TR> f3) => arg => f3(f2(f1(arg)));

        public static Func<T1, T2, TR> Compose<T1, T2, T3, T4, TR>(this Func<T1, T2, T2> f1, Func<T2, T3> f2,
            Func<T3, TR> f3) => (a, b) => f3(f2(f1(a, b)));

        public static Func<T1, T2, T3, TR> Compose<T1, T2, T3, T4, T5, TR>(
            this Func<T1, T2, T3, T4> f1, Func<T4, T5> f2, Func<T5, TR> f3) => (a, b, c) => f3(f2(f1(a, b, c)));

        public static Func<T1, T2, T3, T4, TR> Compose<T1, T2, T3, T4, T5, T6, TR>(
            this Func<T1, T2, T3, T4, T5> f1, Func<T5, T6> f2, Func<T6, TR> f3)
            => (a, b, c, d) => f3(f2(f1(a, b, c, d)));

        public static Func<T1, T2, T3, T5, TR> Compose<T1, T2, T3, T4, T5, TR>(this Func<T1, T2, T3, T4> f1, Func<T4, T5, TR> f2)
          => (a, b, c, d) => f2(f1(a, b, c), d);

        public static Action<T1> Compose<T1, T2>(this Func<T1, T2> func, Action<T2> action) => arg => action(func(arg));


        public static Func<T1, T2, T3, T4, R> Compose<T1, T2, T3, T4, TIR, R>(
          this Func<T1, T2, T3, T4, TIR> func1, Func<TIR, R> func2)
        => (p1, p2, p3, p4) => func2(func1(p1, p2, p3, p4));


        public static Func<T1, T2, T3, T4, T5, R> Compose<T1, T2, T3, T4, T5, TIR, R>(
          this Func<T1, T2, T3, T4, T5, TIR> func1, Func<TIR, R> func2)
        => (p1, p2, p3, p4, p5) => func2(func1(p1, p2, p3, p4, p5));


        public static Func<T1, T2, T3, T4, T5, T6, R> Compose<T1, T2, T3, T4, T5, T6, TIR, R>(
          this Func<T1, T2, T3, T4, T5, T6, TIR> func1, Func<TIR, R> func2)
        => (p1, p2, p3, p4, p5, p6) => func2(func1(p1, p2, p3, p4, p5, p6));


        public static Func<T1, T2, T3, T4, T5, T6, T7, R> Compose<T1, T2, T3, T4, T5, T6, T7, TIR, R>(
          this Func<T1, T2, T3, T4, T5, T6, T7, TIR> func1, Func<TIR, R> func2)
        => (p1, p2, p3, p4, p5, p6, p7) => func2(func1(p1, p2, p3, p4, p5, p6, p7));


        public static Func<T1, T2, T3, T4, T5, T6, T7, T8, R> Compose<T1, T2, T3, T4, T5, T6, T7, T8, TIR, R>(
          this Func<T1, T2, T3, T4, T5, T6, T7, T8, TIR> func1, Func<TIR, R> func2)
        => (p1, p2, p3, p4, p5, p6, p7, p8) => func2(func1(p1, p2, p3, p4, p5, p6, p7, p8));


        public static Func<T1, T2, T3, T4, T5, T6, T7, T8, T9, R> Compose<T1, T2, T3, T4, T5, T6, T7, T8, T9, TIR, R>(
          this Func<T1, T2, T3, T4, T5, T6, T7, T8, T9, TIR> func1, Func<TIR, R> func2)
        => (p1, p2, p3, p4, p5, p6, p7, p8, p9) => func2(func1(p1, p2, p3, p4, p5, p6, p7, p8, p9));


        public static Func<T1, T2, R> Compose<T1, T2, TIR1, TIR2, R>(
          this Func<T1, T2, TIR1> func1, Func<TIR1, TIR2> func2, Func<TIR2, R> func3)
        => (p1, p2) => func3(func2(func1(p1, p2)));



        public static Func<T1, T2, T3, T4, T5, R> Compose<T1, T2, T3, T4, T5, TIR1, TIR2, R>(
          this Func<T1, T2, T3, T4, T5, TIR1> func1, Func<TIR1, TIR2> func2, Func<TIR2, R> func3)
        => (p1, p2, p3, p4, p5) => func3(func2(func1(p1, p2, p3, p4, p5)));


        public static Func<T1, T2, T3, T4, T5, T6, R> Compose<T1, T2, T3, T4, T5, T6, TIR1, TIR2, R>(
          this Func<T1, T2, T3, T4, T5, T6, TIR1> func1, Func<TIR1, TIR2> func2, Func<TIR2, R> func3)
        => (p1, p2, p3, p4, p5, p6) => func3(func2(func1(p1, p2, p3, p4, p5, p6)));


        public static Func<T1, T2, T3, T4, T5, T6, T7, R> Compose<T1, T2, T3, T4, T5, T6, T7, TIR1, TIR2, R>(
          this Func<T1, T2, T3, T4, T5, T6, T7, TIR1> func1, Func<TIR1, TIR2> func2, Func<TIR2, R> func3)
        => (p1, p2, p3, p4, p5, p6, p7) => func3(func2(func1(p1, p2, p3, p4, p5, p6, p7)));


        public static Func<T1, T2, T3, T4, T5, T6, T7, T8, R> Compose<T1, T2, T3, T4, T5, T6, T7, T8, TIR1, TIR2, R>(
          this Func<T1, T2, T3, T4, T5, T6, T7, T8, TIR1> func1, Func<TIR1, TIR2> func2, Func<TIR2, R> func3)
        => (p1, p2, p3, p4, p5, p6, p7, p8) => func3(func2(func1(p1, p2, p3, p4, p5, p6, p7, p8)));


        public static Func<T1, T2, T3, T4, T5, T6, T7, T8, T9, R> Compose<T1, T2, T3, T4, T5, T6, T7, T8, T9, TIR1, TIR2, R>(
          this Func<T1, T2, T3, T4, T5, T6, T7, T8, T9, TIR1> func1, Func<TIR1, TIR2> func2, Func<TIR2, R> func3)
        => (p1, p2, p3, p4, p5, p6, p7, p8, p9) => func3(func2(func1(p1, p2, p3, p4, p5, p6, p7, p8, p9)));

        public static Func<T, R> Compose<T, TIR1, TIR2, TIR3, R>(
          Func<T, TIR1> func1, Func<TIR1, TIR2> func2, Func<TIR2, TIR3> func3, Func<TIR3, R> func4)
        => sourceParam => func4(func3(func2(func1(sourceParam))));


        public static Func<T1, T2, R> Compose<T1, T2, TIR1, TIR2, TIR3, R>(
          this Func<T1, T2, TIR1> func1, Func<TIR1, TIR2> func2, Func<TIR2, TIR3> func3, Func<TIR3, R> func4)
        => (p1, p2) => func4(func3(func2(func1(p1, p2))));


        public static Func<T1, T2, T3, R> Compose<T1, T2, T3, TIR1, TIR2, TIR3, R>(
          this Func<T1, T2, T3, TIR1> func1, Func<TIR1, TIR2> func2, Func<TIR2, TIR3> func3, Func<TIR3, R> func4)
        => (p1, p2, p3) => func4(func3(func2(func1(p1, p2, p3))));


        public static Func<T1, T2, T3, T4, R> Compose<T1, T2, T3, T4, TIR1, TIR2, TIR3, R>(
          this Func<T1, T2, T3, T4, TIR1> func1, Func<TIR1, TIR2> func2, Func<TIR2, TIR3> func3, Func<TIR3, R> func4)
        => (p1, p2, p3, p4) => func4(func3(func2(func1(p1, p2, p3, p4))));


        public static Func<T1, T2, T3, T4, T5, R> Compose<T1, T2, T3, T4, T5, TIR1, TIR2, TIR3, R>(
          this Func<T1, T2, T3, T4, T5, TIR1> func1, Func<TIR1, TIR2> func2, Func<TIR2, TIR3> func3, Func<TIR3, R> func4)
        => (p1, p2, p3, p4, p5) => func4(func3(func2(func1(p1, p2, p3, p4, p5))));


        public static Func<T1, T2, T3, T4, T5, T6, R> Compose<T1, T2, T3, T4, T5, T6, TIR1, TIR2, TIR3, R>(
          this Func<T1, T2, T3, T4, T5, T6, TIR1> func1, Func<TIR1, TIR2> func2, Func<TIR2, TIR3> func3, Func<TIR3, R> func4)
        => (p1, p2, p3, p4, p5, p6) => func4(func3(func2(func1(p1, p2, p3, p4, p5, p6))));


        public static Func<T1, T2, T3, T4, T5, T6, T7, R> Compose<T1, T2, T3, T4, T5, T6, T7, TIR1, TIR2, TIR3, R>(
          this Func<T1, T2, T3, T4, T5, T6, T7, TIR1> func1, Func<TIR1, TIR2> func2, Func<TIR2, TIR3> func3, Func<TIR3, R> func4)
        => (p1, p2, p3, p4, p5, p6, p7) => func4(func3(func2(func1(p1, p2, p3, p4, p5, p6, p7))));


        public static Func<T1, T2, T3, T4, T5, T6, T7, T8, R> Compose<T1, T2, T3, T4, T5, T6, T7, T8, TIR1, TIR2, TIR3, R>(
          this Func<T1, T2, T3, T4, T5, T6, T7, T8, TIR1> func1, Func<TIR1, TIR2> func2, Func<TIR2, TIR3> func3, Func<TIR3, R> func4)
        => (p1, p2, p3, p4, p5, p6, p7, p8) => func4(func3(func2(func1(p1, p2, p3, p4, p5, p6, p7, p8))));


        public static Func<T1, T2, T3, T4, T5, T6, T7, T8, T9, R> Compose<T1, T2, T3, T4, T5, T6, T7, T8, T9, TIR1, TIR2, TIR3, R>(
          this Func<T1, T2, T3, T4, T5, T6, T7, T8, T9, TIR1> func1, Func<TIR1, TIR2> func2, Func<TIR2, TIR3> func3, Func<TIR3, R> func4)
        => (p1, p2, p3, p4, p5, p6, p7, p8, p9) => func4(func3(func2(func1(p1, p2, p3, p4, p5, p6, p7, p8, p9))));


        public static Func<T, R> Compose<T, TIR1, TIR2, TIR3, TIR4, R>(
          Func<T, TIR1> func1, Func<TIR1, TIR2> func2, Func<TIR2, TIR3> func3, Func<TIR3, TIR4> func4, Func<TIR4, R> func5)
        => sourceParam => func5(func4(func3(func2(func1(sourceParam)))));


        public static Func<T1, T2, R> Compose<T1, T2, TIR1, TIR2, TIR3, TIR4, R>(
          this Func<T1, T2, TIR1> func1, Func<TIR1, TIR2> func2, Func<TIR2, TIR3> func3, Func<TIR3, TIR4> func4, Func<TIR4, R> func5)
        => (p1, p2) => func5(func4(func3(func2(func1(p1, p2)))));


        public static Func<T1, T2, T3, R> Compose<T1, T2, T3, TIR1, TIR2, TIR3, TIR4, R>(
          this Func<T1, T2, T3, TIR1> func1, Func<TIR1, TIR2> func2, Func<TIR2, TIR3> func3, Func<TIR3, TIR4> func4, Func<TIR4, R> func5)
        => (p1, p2, p3) => func5(func4(func3(func2(func1(p1, p2, p3)))));


        public static Func<T1, T2, T3, T4, R> Compose<T1, T2, T3, T4, TIR1, TIR2, TIR3, TIR4, R>(
          this Func<T1, T2, T3, T4, TIR1> func1, Func<TIR1, TIR2> func2, Func<TIR2, TIR3> func3, Func<TIR3, TIR4> func4, Func<TIR4, R> func5)
        => (p1, p2, p3, p4) => func5(func4(func3(func2(func1(p1, p2, p3, p4)))));


        public static Func<T1, T2, T3, T4, T5, R> Compose<T1, T2, T3, T4, T5, TIR1, TIR2, TIR3, TIR4, R>(
          this Func<T1, T2, T3, T4, T5, TIR1> func1, Func<TIR1, TIR2> func2, Func<TIR2, TIR3> func3, Func<TIR3, TIR4> func4, Func<TIR4, R> func5)
        => (p1, p2, p3, p4, p5) => func5(func4(func3(func2(func1(p1, p2, p3, p4, p5)))));


        public static Func<T1, T2, T3, T4, T5, T6, R> Compose<T1, T2, T3, T4, T5, T6, TIR1, TIR2, TIR3, TIR4, R>(
          this Func<T1, T2, T3, T4, T5, T6, TIR1> func1, Func<TIR1, TIR2> func2, Func<TIR2, TIR3> func3, Func<TIR3, TIR4> func4, Func<TIR4, R> func5)
        => (p1, p2, p3, p4, p5, p6) => func5(func4(func3(func2(func1(p1, p2, p3, p4, p5, p6)))));


        public static Func<T1, T2, T3, T4, T5, T6, T7, R> Compose<T1, T2, T3, T4, T5, T6, T7, TIR1, TIR2, TIR3, TIR4, R>(
          this Func<T1, T2, T3, T4, T5, T6, T7, TIR1> func1, Func<TIR1, TIR2> func2, Func<TIR2, TIR3> func3, Func<TIR3, TIR4> func4, Func<TIR4, R> func5)
        => (p1, p2, p3, p4, p5, p6, p7) => func5(func4(func3(func2(func1(p1, p2, p3, p4, p5, p6, p7)))));


        public static Func<T1, T2, T3, T4, T5, T6, T7, T8, R> Compose<T1, T2, T3, T4, T5, T6, T7, T8, TIR1, TIR2, TIR3, TIR4, R>(
          this Func<T1, T2, T3, T4, T5, T6, T7, T8, TIR1> func1, Func<TIR1, TIR2> func2, Func<TIR2, TIR3> func3, Func<TIR3, TIR4> func4, Func<TIR4, R> func5)
        => (p1, p2, p3, p4, p5, p6, p7, p8) => func5(func4(func3(func2(func1(p1, p2, p3, p4, p5, p6, p7, p8)))));


        public static Func<T1, T2, T3, T4, T5, T6, T7, T8, T9, R> Compose<T1, T2, T3, T4, T5, T6, T7, T8, T9, TIR1, TIR2, TIR3, TIR4, R>(
          this Func<T1, T2, T3, T4, T5, T6, T7, T8, T9, TIR1> func1, Func<TIR1, TIR2> func2, Func<TIR2, TIR3> func3, Func<TIR3, TIR4> func4, Func<TIR4, R> func5)
        => (p1, p2, p3, p4, p5, p6, p7, p8, p9) => func5(func4(func3(func2(func1(p1, p2, p3, p4, p5, p6, p7, p8, p9)))));


        public static Action<T1, T2> Compose<T1, T2, TIR>(
          this Func<T1, T2, TIR> func1, Action<TIR> action)
        => (p1, p2) => action(func1(p1, p2));


        public static Action<T1, T2, T3> Compose<T1, T2, T3, TIR>(
          this Func<T1, T2, T3, TIR> func1, Action<TIR> action)
        => (p1, p2, p3) => action(func1(p1, p2, p3));


        public static Action<T1, T2, T3, T4> Compose<T1, T2, T3, T4, TIR>(
          this Func<T1, T2, T3, T4, TIR> func1, Action<TIR> action)
        => (p1, p2, p3, p4) => action(func1(p1, p2, p3, p4));


        public static Action<T1, T2, T3, T4, T5> Compose<T1, T2, T3, T4, T5, TIR>(
          this Func<T1, T2, T3, T4, T5, TIR> func1, Action<TIR> action)
        => (p1, p2, p3, p4, p5) => action(func1(p1, p2, p3, p4, p5));


        public static Action<T1, T2, T3, T4, T5, T6> Compose<T1, T2, T3, T4, T5, T6, TIR>(
          this Func<T1, T2, T3, T4, T5, T6, TIR> func1, Action<TIR> action)
        => (p1, p2, p3, p4, p5, p6) => action(func1(p1, p2, p3, p4, p5, p6));


        public static Action<T1, T2, T3, T4, T5, T6, T7> Compose<T1, T2, T3, T4, T5, T6, T7, TIR>(
          this Func<T1, T2, T3, T4, T5, T6, T7, TIR> func1, Action<TIR> action)
        => (p1, p2, p3, p4, p5, p6, p7) => action(func1(p1, p2, p3, p4, p5, p6, p7));


        public static Action<T1, T2, T3, T4, T5, T6, T7, T8> Compose<T1, T2, T3, T4, T5, T6, T7, T8, TIR>(
          this Func<T1, T2, T3, T4, T5, T6, T7, T8, TIR> func1, Action<TIR> action)
        => (p1, p2, p3, p4, p5, p6, p7, p8) => action(func1(p1, p2, p3, p4, p5, p6, p7, p8));


        public static Action<T1, T2, T3, T4, T5, T6, T7, T8, T9> Compose<T1, T2, T3, T4, T5, T6, T7, T8, T9, TIR>(
          this Func<T1, T2, T3, T4, T5, T6, T7, T8, T9, TIR> func1, Action<TIR> action)
        => (p1, p2, p3, p4, p5, p6, p7, p8, p9) => action(func1(p1, p2, p3, p4, p5, p6, p7, p8, p9));


        public static Action<T> Compose<T, TIR1, TIR2>(
          Func<T, TIR1> func1, Func<TIR1, TIR2> func2, Action<TIR2> action)
        => sourceParam => action(func2(func1(sourceParam)));


        public static Action<T1, T2> Compose<T1, T2, TIR1, TIR2>(
          this Func<T1, T2, TIR1> func1, Func<TIR1, TIR2> func2, Action<TIR2> action)
        => (p1, p2) => action(func2(func1(p1, p2)));


        public static Action<T1, T2, T3> Compose<T1, T2, T3, TIR1, TIR2>(
          this Func<T1, T2, T3, TIR1> func1, Func<TIR1, TIR2> func2, Action<TIR2> action)
        => (p1, p2, p3) => action(func2(func1(p1, p2, p3)));


        public static Action<T1, T2, T3, T4> Compose<T1, T2, T3, T4, TIR1, TIR2>(
          this Func<T1, T2, T3, T4, TIR1> func1, Func<TIR1, TIR2> func2, Action<TIR2> action)
        => (p1, p2, p3, p4) => action(func2(func1(p1, p2, p3, p4)));


        public static Action<T1, T2, T3, T4, T5> Compose<T1, T2, T3, T4, T5, TIR1, TIR2>(
          this Func<T1, T2, T3, T4, T5, TIR1> func1, Func<TIR1, TIR2> func2, Action<TIR2> action)
        => (p1, p2, p3, p4, p5) => action(func2(func1(p1, p2, p3, p4, p5)));


        public static Action<T1, T2, T3, T4, T5, T6> Compose<T1, T2, T3, T4, T5, T6, TIR1, TIR2>(
          this Func<T1, T2, T3, T4, T5, T6, TIR1> func1, Func<TIR1, TIR2> func2, Action<TIR2> action)
        => (p1, p2, p3, p4, p5, p6) => action(func2(func1(p1, p2, p3, p4, p5, p6)));


        public static Action<T1, T2, T3, T4, T5, T6, T7> Compose<T1, T2, T3, T4, T5, T6, T7, TIR1, TIR2>(
          this Func<T1, T2, T3, T4, T5, T6, T7, TIR1> func1, Func<TIR1, TIR2> func2, Action<TIR2> action)
        => (p1, p2, p3, p4, p5, p6, p7) => action(func2(func1(p1, p2, p3, p4, p5, p6, p7)));


        public static Action<T1, T2, T3, T4, T5, T6, T7, T8> Compose<T1, T2, T3, T4, T5, T6, T7, T8, TIR1, TIR2>(
          this Func<T1, T2, T3, T4, T5, T6, T7, T8, TIR1> func1, Func<TIR1, TIR2> func2, Action<TIR2> action)
        => (p1, p2, p3, p4, p5, p6, p7, p8) => action(func2(func1(p1, p2, p3, p4, p5, p6, p7, p8)));


        public static Action<T1, T2, T3, T4, T5, T6, T7, T8, T9> Compose<T1, T2, T3, T4, T5, T6, T7, T8, T9, TIR1, TIR2>(
          this Func<T1, T2, T3, T4, T5, T6, T7, T8, T9, TIR1> func1, Func<TIR1, TIR2> func2, Action<TIR2> action)
        => (p1, p2, p3, p4, p5, p6, p7, p8, p9) => action(func2(func1(p1, p2, p3, p4, p5, p6, p7, p8, p9)));

        public static Action<T> Compose<T, TIR1, TIR2, TIR3>(
          Func<T, TIR1> func1, Func<TIR1, TIR2> func2, Func<TIR2, TIR3> func3, Action<TIR3> action)
        => sourceParam => action(func3(func2(func1(sourceParam))));


        public static Action<T1, T2> Compose<T1, T2, TIR1, TIR2, TIR3>(
          this Func<T1, T2, TIR1> func1, Func<TIR1, TIR2> func2, Func<TIR2, TIR3> func3, Action<TIR3> action)
        => (p1, p2) => action(func3(func2(func1(p1, p2))));


        public static Action<T1, T2, T3> Compose<T1, T2, T3, TIR1, TIR2, TIR3>(
          this Func<T1, T2, T3, TIR1> func1, Func<TIR1, TIR2> func2, Func<TIR2, TIR3> func3, Action<TIR3> action)
        => (p1, p2, p3) => action(func3(func2(func1(p1, p2, p3))));


        public static Action<T1, T2, T3, T4> Compose<T1, T2, T3, T4, TIR1, TIR2, TIR3>(
          this Func<T1, T2, T3, T4, TIR1> func1, Func<TIR1, TIR2> func2, Func<TIR2, TIR3> func3, Action<TIR3> action)
        => (p1, p2, p3, p4) => action(func3(func2(func1(p1, p2, p3, p4))));


        public static Action<T1, T2, T3, T4, T5> Compose<T1, T2, T3, T4, T5, TIR1, TIR2, TIR3>(
          this Func<T1, T2, T3, T4, T5, TIR1> func1, Func<TIR1, TIR2> func2, Func<TIR2, TIR3> func3, Action<TIR3> action)
        => (p1, p2, p3, p4, p5) => action(func3(func2(func1(p1, p2, p3, p4, p5))));


        public static Action<T1, T2, T3, T4, T5, T6> Compose<T1, T2, T3, T4, T5, T6, TIR1, TIR2, TIR3>(
          this Func<T1, T2, T3, T4, T5, T6, TIR1> func1, Func<TIR1, TIR2> func2, Func<TIR2, TIR3> func3, Action<TIR3> action)
        => (p1, p2, p3, p4, p5, p6) => action(func3(func2(func1(p1, p2, p3, p4, p5, p6))));


        public static Action<T1, T2, T3, T4, T5, T6, T7> Compose<T1, T2, T3, T4, T5, T6, T7, TIR1, TIR2, TIR3>(
          this Func<T1, T2, T3, T4, T5, T6, T7, TIR1> func1, Func<TIR1, TIR2> func2, Func<TIR2, TIR3> func3, Action<TIR3> action)
        => (p1, p2, p3, p4, p5, p6, p7) => action(func3(func2(func1(p1, p2, p3, p4, p5, p6, p7))));


        public static Action<T1, T2, T3, T4, T5, T6, T7, T8> Compose<T1, T2, T3, T4, T5, T6, T7, T8, TIR1, TIR2, TIR3>(
          this Func<T1, T2, T3, T4, T5, T6, T7, T8, TIR1> func1, Func<TIR1, TIR2> func2, Func<TIR2, TIR3> func3, Action<TIR3> action)
        => (p1, p2, p3, p4, p5, p6, p7, p8) => action(func3(func2(func1(p1, p2, p3, p4, p5, p6, p7, p8))));


        public static Action<T1, T2, T3, T4, T5, T6, T7, T8, T9> Compose<T1, T2, T3, T4, T5, T6, T7, T8, T9, TIR1, TIR2, TIR3>(
          this Func<T1, T2, T3, T4, T5, T6, T7, T8, T9, TIR1> func1, Func<TIR1, TIR2> func2, Func<TIR2, TIR3> func3, Action<TIR3> action)
        => (p1, p2, p3, p4, p5, p6, p7, p8, p9) => action(func3(func2(func1(p1, p2, p3, p4, p5, p6, p7, p8, p9))));


        public static Action<T> Compose<T, TIR1, TIR2, TIR3, TIR4>(
          Func<T, TIR1> func1, Func<TIR1, TIR2> func2, Func<TIR2, TIR3> func3, Func<TIR3, TIR4> func4, Action<TIR4> action)
        => sourceParam => action(func4(func3(func2(func1(sourceParam)))));


        public static Action<T1, T2> Compose<T1, T2, TIR1, TIR2, TIR3, TIR4>(
          this Func<T1, T2, TIR1> func1, Func<TIR1, TIR2> func2, Func<TIR2, TIR3> func3, Func<TIR3, TIR4> func4, Action<TIR4> action)
        => (p1, p2) => action(func4(func3(func2(func1(p1, p2)))));


        public static Action<T1, T2, T3> Compose<T1, T2, T3, TIR1, TIR2, TIR3, TIR4>(
          this Func<T1, T2, T3, TIR1> func1, Func<TIR1, TIR2> func2, Func<TIR2, TIR3> func3, Func<TIR3, TIR4> func4, Action<TIR4> action)
        => (p1, p2, p3) => action(func4(func3(func2(func1(p1, p2, p3)))));


        public static Action<T1, T2, T3, T4> Compose<T1, T2, T3, T4, TIR1, TIR2, TIR3, TIR4>(
          this Func<T1, T2, T3, T4, TIR1> func1, Func<TIR1, TIR2> func2, Func<TIR2, TIR3> func3, Func<TIR3, TIR4> func4, Action<TIR4> action)
        => (p1, p2, p3, p4) => action(func4(func3(func2(func1(p1, p2, p3, p4)))));


        public static Action<T1, T2, T3, T4, T5> Compose<T1, T2, T3, T4, T5, TIR1, TIR2, TIR3, TIR4>(
          this Func<T1, T2, T3, T4, T5, TIR1> func1, Func<TIR1, TIR2> func2, Func<TIR2, TIR3> func3, Func<TIR3, TIR4> func4, Action<TIR4> action)
        => (p1, p2, p3, p4, p5) => action(func4(func3(func2(func1(p1, p2, p3, p4, p5)))));


        public static Action<T1, T2, T3, T4, T5, T6> Compose<T1, T2, T3, T4, T5, T6, TIR1, TIR2, TIR3, TIR4>(
          this Func<T1, T2, T3, T4, T5, T6, TIR1> func1, Func<TIR1, TIR2> func2, Func<TIR2, TIR3> func3, Func<TIR3, TIR4> func4, Action<TIR4> action)
        => (p1, p2, p3, p4, p5, p6) => action(func4(func3(func2(func1(p1, p2, p3, p4, p5, p6)))));


        public static Action<T1, T2, T3, T4, T5, T6, T7> Compose<T1, T2, T3, T4, T5, T6, T7, TIR1, TIR2, TIR3, TIR4>(
          this Func<T1, T2, T3, T4, T5, T6, T7, TIR1> func1, Func<TIR1, TIR2> func2, Func<TIR2, TIR3> func3, Func<TIR3, TIR4> func4, Action<TIR4> action)
        => (p1, p2, p3, p4, p5, p6, p7) => action(func4(func3(func2(func1(p1, p2, p3, p4, p5, p6, p7)))));


        public static Action<T1, T2, T3, T4, T5, T6, T7, T8> Compose<T1, T2, T3, T4, T5, T6, T7, T8, TIR1, TIR2, TIR3, TIR4>(
          this Func<T1, T2, T3, T4, T5, T6, T7, T8, TIR1> func1, Func<TIR1, TIR2> func2, Func<TIR2, TIR3> func3, Func<TIR3, TIR4> func4, Action<TIR4> action)
        => (p1, p2, p3, p4, p5, p6, p7, p8) => action(func4(func3(func2(func1(p1, p2, p3, p4, p5, p6, p7, p8)))));


        public static Action<T1, T2, T3, T4, T5, T6, T7, T8, T9> Compose<T1, T2, T3, T4, T5, T6, T7, T8, T9, TIR1, TIR2, TIR3, TIR4>(
          this Func<T1, T2, T3, T4, T5, T6, T7, T8, T9, TIR1> func1, Func<TIR1, TIR2> func2, Func<TIR2, TIR3> func3, Func<TIR3, TIR4> func4, Action<TIR4> action)
        => (p1, p2, p3, p4, p5, p6, p7, p8, p9) => action(func4(func3(func2(func1(p1, p2, p3, p4, p5, p6, p7, p8, p9)))));

    }
}