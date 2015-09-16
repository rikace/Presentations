## LambdaMicrobenchmarking: microbenchmarking tool for C# and F#.
[![Build status](https://ci.appveyor.com/api/projects/status/kk8gk4cw9lre9fp7/branch/master?svg=true)](https://ci.appveyor.com/project/biboudis/lambdamicrobenchmarking/branch/master)
[![NuGet](https://img.shields.io/nuget/v/lambdamicrobenchmarking.svg?style=flat)](https://www.nuget.org/packages/LambdaMicrobenchmarking/)
[![NuGet total](https://img.shields.io/nuget/dt/LambdaMicrobenchmarking.svg?style=flat)](https://www.nuget.org/packages/LambdaMicrobenchmarking/)

LambdaMicrobenchmarking is a library for microbenchmarking orchestration in C# and F#. The programer can measure the execution performance of thunks (lambdas with no arguments).
```C#
Func<long> sumLinq       = () => v.Sum();
Func<long> sumSqLinq     = () => v.Select(x => x * x).Sum();
Func<long> sumSqEvenLinq = () => v.Where(x => x % 2 == 0).Select(x => x * x).Sum();
Func<long> cartLinq      = () => (from x in vHi
                                  from y in vLow
                                  select x * y).Sum();
Script.Of("sumLinq", sumLinq)
      .Of("sumSqLinq", sumSqLinq)
      .Of("sumSqEvensLinq", sumSqEvenLinq)
      .Of("cartLinq", cartLinq)
      .WithHead()
      .RunAll();
```
The corresponding script for F#:
```F#
Script.Of("sumLinq", Func<int64> sumLinq)
      .Of("sumOfSquaresLinq", Func<int64> sumSqLinq)
      .Of("sumOfSquaresEvenLinq", Func<int64> sumSqEvenLinq)
      .Of("cartLinq", Func<int64> cartLinq)
      .WithHead()
      .RunAll |> ignore
```
Sample output:
```
Benchmark          Mean      Mean-Error  Sdev  Unit
sumLinq          98.324           0.543 0.359 ms/op
sumSqLinq       224.831           1.804 1.193 ms/op
sumSqEvenLinq    82,789           5,729 3,789 ms/op
cartLinq        200,775           6,477 4,284 ms/op
```

\* the statistics part is inspired by [JMH](http://openjdk.java.net/projects/code-tools/jmh/) ([AbstractStatistics](http://hg.openjdk.java.net/code-tools/jmh/file/75f8b23444f6/jmh-core/src/main/java/org/openjdk/jmh/util/internal/AbstractStatistics.java), [ListStatistics](http://hg.openjdk.java.net/code-tools/jmh/file/75f8b23444f6/jmh-core/src/main/java/org/openjdk/jmh/util/internal/ListStatistics.java)).
 
### References
* [Clash of the Lambdas](http://biboudis.github.io/clashofthelambdas/)
* [Microbenchmarks in Java and C#](http://www.itu.dk/people/sestoft/papers/benchmarking.pdf)
* [JMH](http://openjdk.java.net/projects/code-tools/jmh/)

### Contributing
Sending PRs and participating in the discussion for improving LM is **highly** encouraged. Our goal is to create a tool for .NET that produces valid results, giving real control over the intented experiments.
