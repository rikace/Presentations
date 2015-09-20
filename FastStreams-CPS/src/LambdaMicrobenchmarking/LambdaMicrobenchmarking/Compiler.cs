using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace LambdaMicrobenchmarking
{
    class Compiler
    {
        [MethodImpl(MethodImplOptions.NoOptimization|MethodImplOptions.NoInlining)]
        static public T ConsumeValue<T>(T dummy)
        {
            return dummy;
        }
    }
}
