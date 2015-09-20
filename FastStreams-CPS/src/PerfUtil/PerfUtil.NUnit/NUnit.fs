namespace PerfUtil.NUnit

    open NUnit.Framework

    open PerfUtil

    module private Utils =

        // add single quotes if text contains whitespace
        let quoteText (text : string) =
            if text |> Seq.exists System.Char.IsWhiteSpace then
                sprintf "'%s'" text
            else
                text

    [<AbstractClass>]
    [<TestFixture>]
    /// Inheriting this class in an assembly defines a dynamic NUnit test fixture.
    type NUnitPerf<'Impl when 'Impl :> ITestable> () =

        /// specifies the performance testbed to be used.
        abstract PerfTester : PerformanceTester<'Impl>
        /// specifies the performance tests to be tested.
        abstract PerfTests : PerfTest<'Impl> list

        member internal u.GetTestCases () = 
            u.PerfTests |> Seq.map (fun t -> TestCaseData(t).SetName(Utils.quoteText t.Id))

        [<Test ; TestCaseSource("GetTestCases")>]
        member u.PerformanceTests(test : PerfTest<'Impl>) = u.PerfTester.RunTest test
