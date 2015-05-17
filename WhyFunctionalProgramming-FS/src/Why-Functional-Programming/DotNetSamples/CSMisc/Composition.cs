using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace CSMisc
{
    public class Composition
    {
        int CalcB(int a)
        {
            return a * 3;
        }
        int CalcC(int b)
        {
            return b + 27;
        }

        int CalcCfromA(int a)
        {
            return CalcC(CalcB(a));
        }


        public void Test()
        {

            int a = 10;
            int b = CalcB(a); int c = CalcC(b);

            Func<int, int> calcFunc = x => CalcC(CalcB(x));

            Func<int, int> calcM = x => x * 3;
            Func<int, int> calcA = x => x + 27;
            var calcCFromA = ComposeEx.Compose(calcM, calcA); // Alternatively:
            var calcCFromA_ = calcM.Compose(calcA);

        }
    }

    public static class ComposeEx
    {
        public static Func<TSource, TEndResult> Compose<TSource, TIntermediateResult, TEndResult>(
this Func<TSource, TIntermediateResult> func1,
Func<TIntermediateResult, TEndResult> func2)
        {
            return sourceParam => func2(func1(sourceParam));
        }
    }
}

