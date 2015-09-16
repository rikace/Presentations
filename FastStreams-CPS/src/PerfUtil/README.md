##PerfUtil

A collection of tools and abstractions for helping performance tests.
Two main operation modes are provided:
* Comparison of a given implementation against others.
* Comparison of current implementation against a history of past performance tests.

A NuGet package is available [here](https://www.nuget.org/packages/PerfUtil/).

###Basic Usage

```fsharp
open PerfUtil

let result = Benchmark.Run (repeat 100 (fun () -> Thread.Sleep 10))

val result : PerfResult = {TestId = "";
                           SessionId = "";
                           Date = 2/12/2013 7:30:01 pm;
                           Error = null;
                           Elapsed = 00:00:00.9998810;
                           CpuTime = 00:00:00;
                           GcDelta = [0; 0; 0];}
```

###Comparing implementations

Defining a test context:
```fsharp
type IOperation =
    inherit ITestable
    abstract Run : unit -> unit

let dummy name (interval:int) = 
    {
        new IOperation with
            member __.Name = name
            member __.Run () = System.Threading.Thread.Sleep(interval)
    }

let tested = dummy "foo" 10

```
#### Testing against other implementations
```fsharp
let testBed = new OtherImplemantationTester<IOperation>(tested, [dummy "bar" 5 ; dummy "baz" 20 ])

testBed.Test "test 0" (repeat 100 (fun o -> o.Run()))
// Output
// 'test 0': foo was 0.50x faster and 1.00x more memory efficient than bar.
// 'test 0': foo was 2.00x faster and 1.00x more memory efficient than baz.
```
#### Testing against past test runs
```fsharp
let test = new PastImplementationTester<IOperation>(tested, Version(0,3), historyFile = "persist.xml")

test.Test "test 0" (repeat 100 (fun o -> o.Run()))
// Output
// 'test 0': 'foo v.0.3' was 1.00x faster and 1.00x more memory efficient than 'foo v.0.1'.
// 'test 0': 'foo v.0.3' was 1.00x faster and 1.00x more memory efficient than 'foo v.0.2'.

// append current results to history file
test.PersistCurrentResults()
```
#### Defining abstract performance tests

In PerfUtil, an abstract performance test can be represented with the record:
```fsharp
type PerfTest<IOperation> =
    {
        Id : string
        Test : IOPeration -> unit
    }

```
Performance tests can be declared in the following manner:
```fsharp
type Tests =

    [<PerfTest>]
    static member ``Test 1`` (o : IOperation) = o.Run ()

    [<PerfTest>]
    static member ``Test 2`` (o : IOperation) = o |> repeat 100 (fun o -> o.Run ())


let tests = PerfTest<IOperation>.OfType<Tests> ()
// val tests : PerfTest<IOperation> list =
//   [{Id = "Tests.Test 1";
//     Test = <fun:Wrap@90>;}; {Id = "Tests.Test 2";
//                              Test = <fun:Wrap@90>;}]
```
Tests can then be run with a concrete performance tester like so:
```fsharp
tests |> PerfTest.run (fun () -> new PastImplementationTester<IOperation>(...))
```
It is possible to define performance tests in F# modules using the following technique:
```fsharp
module Tests =

    type Marker = class end

    [<PerfTest>]
    let ``Test 0`` (o : IOperation) = o.Run ()


let test = PerfTest<IOperation>.OfModuleMarker<Tests.Marker> () |> List.head

```

#### NUnit Support

A collection of performance tests can be used to define NUnit tests.
To do so, simply place a concrete instance of the `NUnitPerf` abstract class
in your assembly.
```fsharp
[<AbstractClass>]
[<TestFixture>]
type NUnitPerf<'Impl when 'Impl :> ITestable> () =
    abstract PerfTester : PerformanceTester<'Impl>
    abstract PerfTests : PerfTest<'Impl> list
```

#### Plotting Results

Using `FSharp.Charting`, the following code provides a way to plot test results:
```fsharp
open FSharp.Charting
open PerfUtil

// simple plot function
let plot yaxis (metric : PerfResult -> float) (results : PerfResult list) =
    let values = results |> List.choose (fun r -> if r.HasFailed then None else Some (r.SessionId, metric r))
    let name = results |> List.tryPick (fun r -> Some r.TestId)
    let ch = Chart.Bar(values, ?Name = name, ?Title = name, YTitle = yaxis)
    ch.ShowChart()


// read performance tests from 'Tests' module and run them
let results =
    PerfTest<IOperation>.OfModuleMarker<Tests.Marker>()
    |> PerfTest.run SerializerComparer.Create

// plot everything
TestSession.groupByTest results
|> Map.iter (fun _ r -> plot "milliseconds" (fun r -> r.Elapsed.TotalMilliseconds) r)

```

#### Case Study

For more in-depth examples, I have included a simple performance testing implementation 
for the `FsPickler` serializer, which can be found in the `PerfUtil.CaseStudy` project.
