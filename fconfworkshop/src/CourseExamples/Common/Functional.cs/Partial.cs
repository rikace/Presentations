using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Functional
{
    public static partial class Functional
    {
        public static Func<T2, R> Partial<T1, T2, R>(this Func<T1, T2, R> func, T1 a) => (T2 b) => func(a, b);

        public static Func<T3, R> Partial<T1, T2, T3, R>(this Func<T1, T2, T3, R> func, T1 a, T2 b)
            => (T3 c) => func(a, b, c);

        public static Func<T2, T3, R> Partial<T1, T2, T3, R>(this Func<T1, T2, T3, R> func, T1 a)
            => (T2 b, T3 c) => func(a, b, c);

        public static Func<T2, T3, T4, TR> Partial<T1, T2, T3, T4, TR>(this Func<T1, T2, T3, T4, TR> func, T1 arg)
            => (arg2, arg3, arg4) => func(arg, arg2, arg3, arg4);

        public static Func<T3, T4, TR> Partial<T1, T2, T3, T4, TR>(this Func<T1, T2, T3, T4, TR> func, T1 arg, T2 arg2)
            => (arg3, arg4) => func(arg, arg2, arg3, arg4);

        public static Func<T4, TR> Partial<T1, T2, T3, T4, TR>(this Func<T1, T2, T3, T4, TR> func, T1 arg, T2 arg2,
            T3 arg3)
            => arg4 => func(arg, arg2, arg3, arg4);

        public static Func<T2, T3, T4, T5, TR> Partial<T1, T2, T3, T4, T5, TR>(this Func<T1, T2, T3, T4, T5, TR> func,
            T1 arg) => (arg2, arg3, arg4, arg5) => func(arg, arg2, arg3, arg4, arg5);

        public static Func<T3, T4, T5, TR> Partial<T1, T2, T3, T4, T5, TR>(this Func<T1, T2, T3, T4, T5, TR> func,
            T1 arg, T2 arg2) => (arg3, arg4, arg5) => func(arg, arg2, arg3, arg4, arg5);

        public static Func<T4, T5, TR> Partial<T1, T2, T3, T4, T5, TR>(this Func<T1, T2, T3, T4, T5, TR> func, T1 arg,
            T2 arg2, T3 arg3) => (arg4, arg5) => func(arg, arg2, arg3, arg4, arg5);

        public static Func<T5, TR> Partial<T1, T2, T3, T4, T5, TR>(this Func<T1, T2, T3, T4, T5, TR> func, T1 arg,
            T2 arg2, T3 arg3, T4 arg4) => arg5 => func(arg, arg2, arg3, arg4, arg5);

        public static Func<T2, T3, T4, T5, T6, TR> Partial<T1, T2, T3, T4, T5, T6, TR>(Func<T1, T2, T3, T4, T5, T6, TR> function, T1 arg)
        => (arg2, arg3, arg4, arg5, arg6) => function(arg, arg2, arg3, arg4, arg5, arg6);


        public static Func<T3, T4, T5, T6, TR> Partial<T1, T2, T3, T4, T5, T6, TR>(Func<T1, T2, T3, T4, T5, T6, TR> function, T1 arg, T2 arg2)
        => (arg3, arg4, arg5, arg6) => function(arg, arg2, arg3, arg4, arg5, arg6);


        public static Func<T4, T5, T6, TR> Partial<T1, T2, T3, T4, T5, T6, TR>(Func<T1, T2, T3, T4, T5, T6, TR> function, T1 arg, T2 arg2, T3 arg3)
        => (arg4, arg5, arg6) => function(arg, arg2, arg3, arg4, arg5, arg6);


        public static Func<T5, T6, TR> Partial<T1, T2, T3, T4, T5, T6, TR>(Func<T1, T2, T3, T4, T5, T6, TR> function, T1 arg, T2 arg2, T3 arg3, T4 arg4)
        => (arg5, arg6) => function(arg, arg2, arg3, arg4, arg5, arg6);


        public static Func<T6, TR> Partial<T1, T2, T3, T4, T5, T6, TR>(Func<T1, T2, T3, T4, T5, T6, TR> function, T1 arg, T2 arg2, T3 arg3, T4 arg4, T5 arg5)
        => arg6 => function(arg, arg2, arg3, arg4, arg5, arg6);


        public static Func<T2, T3, T4, T5, T6, T7, TR> Partial<T1, T2, T3, T4, T5, T6, T7, TR>(Func<T1, T2, T3, T4, T5, T6, T7, TR> function, T1 arg)
        => (arg2, arg3, arg4, arg5, arg6, arg7) => function(arg, arg2, arg3, arg4, arg5, arg6, arg7);


        public static Func<T3, T4, T5, T6, T7, TR> Partial<T1, T2, T3, T4, T5, T6, T7, TR>(Func<T1, T2, T3, T4, T5, T6, T7, TR> function, T1 arg, T2 arg2)
        => (arg3, arg4, arg5, arg6, arg7) => function(arg, arg2, arg3, arg4, arg5, arg6, arg7);


        public static Func<T4, T5, T6, T7, TR> Partial<T1, T2, T3, T4, T5, T6, T7, TR>(Func<T1, T2, T3, T4, T5, T6, T7, TR> function, T1 arg, T2 arg2, T3 arg3)
        => (arg4, arg5, arg6, arg7) => function(arg, arg2, arg3, arg4, arg5, arg6, arg7);


        public static Func<T5, T6, T7, TR> Partial<T1, T2, T3, T4, T5, T6, T7, TR>(Func<T1, T2, T3, T4, T5, T6, T7, TR> function, T1 arg, T2 arg2, T3 arg3, T4 arg4)
        => (arg5, arg6, arg7) => function(arg, arg2, arg3, arg4, arg5, arg6, arg7);


        public static Func<T6, T7, TR> Partial<T1, T2, T3, T4, T5, T6, T7, TR>(Func<T1, T2, T3, T4, T5, T6, T7, TR> function, T1 arg, T2 arg2, T3 arg3, T4 arg4, T5 arg5)
        => (arg6, arg7) => function(arg, arg2, arg3, arg4, arg5, arg6, arg7);


        public static Func<T7, TR> Partial<T1, T2, T3, T4, T5, T6, T7, TR>(Func<T1, T2, T3, T4, T5, T6, T7, TR> function, T1 arg, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6)
        => arg7 => function(arg, arg2, arg3, arg4, arg5, arg6, arg7);

        public static Func<T2, T3, T4, T5, T6, T7, T8, TR> Partial<T1, T2, T3, T4, T5, T6, T7, T8, TR>(Func<T1, T2, T3, T4, T5, T6, T7, T8, TR> function, T1 arg)
        => (arg2, arg3, arg4, arg5, arg6, arg7, arg8) => function(arg, arg2, arg3, arg4, arg5, arg6, arg7, arg8);


        public static Func<T3, T4, T5, T6, T7, T8, TR> Partial<T1, T2, T3, T4, T5, T6, T7, T8, TR>(Func<T1, T2, T3, T4, T5, T6, T7, T8, TR> function, T1 arg, T2 arg2)
        => (arg3, arg4, arg5, arg6, arg7, arg8) => function(arg, arg2, arg3, arg4, arg5, arg6, arg7, arg8);


        public static Func<T4, T5, T6, T7, T8, TR> Partial<T1, T2, T3, T4, T5, T6, T7, T8, TR>(Func<T1, T2, T3, T4, T5, T6, T7, T8, TR> function, T1 arg, T2 arg2, T3 arg3)
        => (arg4, arg5, arg6, arg7, arg8) => function(arg, arg2, arg3, arg4, arg5, arg6, arg7, arg8);


        public static Func<T5, T6, T7, T8, TR> Partial<T1, T2, T3, T4, T5, T6, T7, T8, TR>(Func<T1, T2, T3, T4, T5, T6, T7, T8, TR> function, T1 arg, T2 arg2, T3 arg3, T4 arg4)
        => (arg5, arg6, arg7, arg8) => function(arg, arg2, arg3, arg4, arg5, arg6, arg7, arg8);


        public static Func<T6, T7, T8, TR> Partial<T1, T2, T3, T4, T5, T6, T7, T8, TR>(Func<T1, T2, T3, T4, T5, T6, T7, T8, TR> function, T1 arg, T2 arg2, T3 arg3, T4 arg4, T5 arg5)
        => (arg6, arg7, arg8) => function(arg, arg2, arg3, arg4, arg5, arg6, arg7, arg8);


        public static Func<T7, T8, TR> Partial<T1, T2, T3, T4, T5, T6, T7, T8, TR>(Func<T1, T2, T3, T4, T5, T6, T7, T8, TR> function, T1 arg, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6)
        => (arg7, arg8) => function(arg, arg2, arg3, arg4, arg5, arg6, arg7, arg8);


        public static Func<T8, TR> Partial<T1, T2, T3, T4, T5, T6, T7, T8, TR>(Func<T1, T2, T3, T4, T5, T6, T7, T8, TR> function, T1 arg, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7)
        => arg8 => function(arg, arg2, arg3, arg4, arg5, arg6, arg7, arg8);

        public static Func<T2, T3, T4, T5, T6, T7, T8, T9, TR> Partial<T1, T2, T3, T4, T5, T6, T7, T8, T9, TR>(Func<T1, T2, T3, T4, T5, T6, T7, T8, T9, TR> function, T1 arg)
        => (arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9) => function(arg, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9);


        public static Func<T3, T4, T5, T6, T7, T8, T9, TR> Partial<T1, T2, T3, T4, T5, T6, T7, T8, T9, TR>(Func<T1, T2, T3, T4, T5, T6, T7, T8, T9, TR> function, T1 arg, T2 arg2)
        => (arg3, arg4, arg5, arg6, arg7, arg8, arg9) => function(arg, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9);


        public static Func<T4, T5, T6, T7, T8, T9, TR> Partial<T1, T2, T3, T4, T5, T6, T7, T8, T9, TR>(Func<T1, T2, T3, T4, T5, T6, T7, T8, T9, TR> function, T1 arg, T2 arg2, T3 arg3)
        => (arg4, arg5, arg6, arg7, arg8, arg9) => function(arg, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9);


        public static Func<T5, T6, T7, T8, T9, TR> Partial<T1, T2, T3, T4, T5, T6, T7, T8, T9, TR>(Func<T1, T2, T3, T4, T5, T6, T7, T8, T9, TR> function, T1 arg, T2 arg2, T3 arg3, T4 arg4)
        => (arg5, arg6, arg7, arg8, arg9) => function(arg, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9);


        public static Func<T6, T7, T8, T9, TR> Partial<T1, T2, T3, T4, T5, T6, T7, T8, T9, TR>(Func<T1, T2, T3, T4, T5, T6, T7, T8, T9, TR> function, T1 arg, T2 arg2, T3 arg3, T4 arg4, T5 arg5)
        => (arg6, arg7, arg8, arg9) => function(arg, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9);


        public static Func<T7, T8, T9, TR> Partial<T1, T2, T3, T4, T5, T6, T7, T8, T9, TR>(Func<T1, T2, T3, T4, T5, T6, T7, T8, T9, TR> function, T1 arg, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6)
        => (arg7, arg8, arg9) => function(arg, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9);


        public static Func<T8, T9, TR> Partial<T1, T2, T3, T4, T5, T6, T7, T8, T9, TR>(Func<T1, T2, T3, T4, T5, T6, T7, T8, T9, TR> function, T1 arg, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7)
        => (arg8, arg9) => function(arg, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9);


        public static Func<T9, TR> Partial<T1, T2, T3, T4, T5, T6, T7, T8, T9, TR>(Func<T1, T2, T3, T4, T5, T6, T7, T8, T9, TR> function, T1 arg, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8)
        => arg9 => function(arg, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9);

        public static Action<T2> Partial<T1, T2>(Action<T1, T2> action, T1 arg)
        => arg2 => action(arg, arg2);


        public static Action<T2, T3> Partial<T1, T2, T3>(Action<T1, T2, T3> action, T1 arg)
        => (arg2, arg3) => action(arg, arg2, arg3);


        public static Action<T3> Partial<T1, T2, T3>(Action<T1, T2, T3> action, T1 arg, T2 arg2)
        => arg3 => action(arg, arg2, arg3);


        public static Action<T2, T3, T4> Partial<T1, T2, T3, T4>(Action<T1, T2, T3, T4> action, T1 arg)
        => (arg2, arg3, arg4) => action(arg, arg2, arg3, arg4);


        public static Action<T3, T4> Partial<T1, T2, T3, T4>(Action<T1, T2, T3, T4> action, T1 arg, T2 arg2)
        => (arg3, arg4) => action(arg, arg2, arg3, arg4);


        public static Action<T4> Partial<T1, T2, T3, T4>(Action<T1, T2, T3, T4> action, T1 arg, T2 arg2, T3 arg3)
        => arg4 => action(arg, arg2, arg3, arg4);


        public static Action<T2, T3, T4, T5> Partial<T1, T2, T3, T4, T5>(Action<T1, T2, T3, T4, T5> action, T1 arg)
        => (arg2, arg3, arg4, arg5) => action(arg, arg2, arg3, arg4, arg5);


        public static Action<T3, T4, T5> Partial<T1, T2, T3, T4, T5>(Action<T1, T2, T3, T4, T5> action, T1 arg, T2 arg2)
        => (arg3, arg4, arg5) => action(arg, arg2, arg3, arg4, arg5);


        public static Action<T4, T5> Partial<T1, T2, T3, T4, T5>(Action<T1, T2, T3, T4, T5> action, T1 arg, T2 arg2, T3 arg3)
        => (arg4, arg5) => action(arg, arg2, arg3, arg4, arg5);


        public static Action<T5> Partial<T1, T2, T3, T4, T5>(Action<T1, T2, T3, T4, T5> action, T1 arg, T2 arg2, T3 arg3, T4 arg4)
        => arg5 => action(arg, arg2, arg3, arg4, arg5);


        public static Action<T2, T3, T4, T5, T6> Partial<T1, T2, T3, T4, T5, T6>(Action<T1, T2, T3, T4, T5, T6> action, T1 arg)
        => (arg2, arg3, arg4, arg5, arg6) => action(arg, arg2, arg3, arg4, arg5, arg6);


        public static Action<T3, T4, T5, T6> Partial<T1, T2, T3, T4, T5, T6>(Action<T1, T2, T3, T4, T5, T6> action, T1 arg, T2 arg2)
        => (arg3, arg4, arg5, arg6) => action(arg, arg2, arg3, arg4, arg5, arg6);


        public static Action<T4, T5, T6> Partial<T1, T2, T3, T4, T5, T6>(Action<T1, T2, T3, T4, T5, T6> action, T1 arg, T2 arg2, T3 arg3)
        => (arg4, arg5, arg6) => action(arg, arg2, arg3, arg4, arg5, arg6);


        public static Action<T5, T6> Partial<T1, T2, T3, T4, T5, T6>(Action<T1, T2, T3, T4, T5, T6> action, T1 arg, T2 arg2, T3 arg3, T4 arg4)
        => (arg5, arg6) => action(arg, arg2, arg3, arg4, arg5, arg6);


        public static Action<T6> Partial<T1, T2, T3, T4, T5, T6>(Action<T1, T2, T3, T4, T5, T6> action, T1 arg, T2 arg2, T3 arg3, T4 arg4, T5 arg5)
        => arg6 => action(arg, arg2, arg3, arg4, arg5, arg6);


        public static Action<T2, T3, T4, T5, T6, T7> Partial<T1, T2, T3, T4, T5, T6, T7>(Action<T1, T2, T3, T4, T5, T6, T7> action, T1 arg)
        => (arg2, arg3, arg4, arg5, arg6, arg7) => action(arg, arg2, arg3, arg4, arg5, arg6, arg7);


        public static Action<T3, T4, T5, T6, T7> Partial<T1, T2, T3, T4, T5, T6, T7>(Action<T1, T2, T3, T4, T5, T6, T7> action, T1 arg, T2 arg2)
        => (arg3, arg4, arg5, arg6, arg7) => action(arg, arg2, arg3, arg4, arg5, arg6, arg7);


        public static Action<T4, T5, T6, T7> Partial<T1, T2, T3, T4, T5, T6, T7>(Action<T1, T2, T3, T4, T5, T6, T7> action, T1 arg, T2 arg2, T3 arg3)
        => (arg4, arg5, arg6, arg7) => action(arg, arg2, arg3, arg4, arg5, arg6, arg7);


        public static Action<T5, T6, T7> Partial<T1, T2, T3, T4, T5, T6, T7>(Action<T1, T2, T3, T4, T5, T6, T7> action, T1 arg, T2 arg2, T3 arg3, T4 arg4)
        => (arg5, arg6, arg7) => action(arg, arg2, arg3, arg4, arg5, arg6, arg7);


        public static Action<T6, T7> Partial<T1, T2, T3, T4, T5, T6, T7>(Action<T1, T2, T3, T4, T5, T6, T7> action, T1 arg, T2 arg2, T3 arg3, T4 arg4, T5 arg5)
        => (arg6, arg7) => action(arg, arg2, arg3, arg4, arg5, arg6, arg7);


        public static Action<T7> Partial<T1, T2, T3, T4, T5, T6, T7>(Action<T1, T2, T3, T4, T5, T6, T7> action, T1 arg, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6)
        => arg7 => action(arg, arg2, arg3, arg4, arg5, arg6, arg7);



        public static Action<T2, T3, T4, T5, T6, T7, T8> Partial<T1, T2, T3, T4, T5, T6, T7, T8>(Action<T1, T2, T3, T4, T5, T6, T7, T8> action, T1 arg)
        => (arg2, arg3, arg4, arg5, arg6, arg7, arg8) => action(arg, arg2, arg3, arg4, arg5, arg6, arg7, arg8);


        public static Action<T3, T4, T5, T6, T7, T8> Partial<T1, T2, T3, T4, T5, T6, T7, T8>(Action<T1, T2, T3, T4, T5, T6, T7, T8> action, T1 arg, T2 arg2)
        => (arg3, arg4, arg5, arg6, arg7, arg8) => action(arg, arg2, arg3, arg4, arg5, arg6, arg7, arg8);


        public static Action<T4, T5, T6, T7, T8> Partial<T1, T2, T3, T4, T5, T6, T7, T8>(Action<T1, T2, T3, T4, T5, T6, T7, T8> action, T1 arg, T2 arg2, T3 arg3)
        => (arg4, arg5, arg6, arg7, arg8) => action(arg, arg2, arg3, arg4, arg5, arg6, arg7, arg8);


        public static Action<T5, T6, T7, T8> Partial<T1, T2, T3, T4, T5, T6, T7, T8>(Action<T1, T2, T3, T4, T5, T6, T7, T8> action, T1 arg, T2 arg2, T3 arg3, T4 arg4)
        => (arg5, arg6, arg7, arg8) => action(arg, arg2, arg3, arg4, arg5, arg6, arg7, arg8);


        public static Action<T6, T7, T8> Partial<T1, T2, T3, T4, T5, T6, T7, T8>(Action<T1, T2, T3, T4, T5, T6, T7, T8> action, T1 arg, T2 arg2, T3 arg3, T4 arg4, T5 arg5)
        => (arg6, arg7, arg8) => action(arg, arg2, arg3, arg4, arg5, arg6, arg7, arg8);


        public static Action<T7, T8> Partial<T1, T2, T3, T4, T5, T6, T7, T8>(Action<T1, T2, T3, T4, T5, T6, T7, T8> action, T1 arg, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6)
        => (arg7, arg8) => action(arg, arg2, arg3, arg4, arg5, arg6, arg7, arg8);


        public static Action<T8> Partial<T1, T2, T3, T4, T5, T6, T7, T8>(Action<T1, T2, T3, T4, T5, T6, T7, T8> action, T1 arg, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7)
        => arg8 => action(arg, arg2, arg3, arg4, arg5, arg6, arg7, arg8);


        public static Action<T2, T3, T4, T5, T6, T7, T8, T9> Partial<T1, T2, T3, T4, T5, T6, T7, T8, T9>(Action<T1, T2, T3, T4, T5, T6, T7, T8, T9> action, T1 arg)
        => (arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9) => action(arg, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9);


        public static Action<T3, T4, T5, T6, T7, T8, T9> Partial<T1, T2, T3, T4, T5, T6, T7, T8, T9>(Action<T1, T2, T3, T4, T5, T6, T7, T8, T9> action, T1 arg, T2 arg2)
        => (arg3, arg4, arg5, arg6, arg7, arg8, arg9) => action(arg, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9);


        public static Action<T4, T5, T6, T7, T8, T9> Partial<T1, T2, T3, T4, T5, T6, T7, T8, T9>(Action<T1, T2, T3, T4, T5, T6, T7, T8, T9> action, T1 arg, T2 arg2, T3 arg3)
        => (arg4, arg5, arg6, arg7, arg8, arg9) => action(arg, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9);


        public static Action<T5, T6, T7, T8, T9> Partial<T1, T2, T3, T4, T5, T6, T7, T8, T9>(Action<T1, T2, T3, T4, T5, T6, T7, T8, T9> action, T1 arg, T2 arg2, T3 arg3, T4 arg4)
        => (arg5, arg6, arg7, arg8, arg9) => action(arg, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9);


        public static Action<T6, T7, T8, T9> Partial<T1, T2, T3, T4, T5, T6, T7, T8, T9>(Action<T1, T2, T3, T4, T5, T6, T7, T8, T9> action, T1 arg, T2 arg2, T3 arg3, T4 arg4, T5 arg5)
        => (arg6, arg7, arg8, arg9) => action(arg, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9);


        public static Action<T7, T8, T9> Partial<T1, T2, T3, T4, T5, T6, T7, T8, T9>(Action<T1, T2, T3, T4, T5, T6, T7, T8, T9> action, T1 arg, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6)
        => (arg7, arg8, arg9) => action(arg, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9);


        public static Action<T8, T9> Partial<T1, T2, T3, T4, T5, T6, T7, T8, T9>(Action<T1, T2, T3, T4, T5, T6, T7, T8, T9> action, T1 arg, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7)
        => (arg8, arg9) => action(arg, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9);


        public static Action<T9> Partial<T1, T2, T3, T4, T5, T6, T7, T8, T9>(Action<T1, T2, T3, T4, T5, T6, T7, T8, T9> action, T1 arg, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8)
        => arg9 => action(arg, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9);
    }
}

