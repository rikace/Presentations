namespace PerfUtil

    open System
    open System.Collections.Generic

    open PerfUtil.Utils

    /// <summary>Compares given implementation performance against a collection of other implementations.</summary>
    /// <param name="testedImpl">Implementation under test.</param>
    /// <param name="otherImpls">Secondary implementations to be compared against.</param>
    /// <param name="comparer">Specifies a custom performance comparer. Default to the TimeComparer.</param>
    /// <param name="warmup">Perform a warmup run before attempting benchmark. Defaults to false.</param>
    /// <param name="verbose">Print performance results to stdout.</param>
    /// <param name="throwOnError">Raise an exception if performance comparison fails. Defaults to false.</param>
    type ImplementationComparer<'Testable when 'Testable :> ITestable>
        (testedImpl : 'Testable, otherImpls : 'Testable list, ?comparer : IPerformanceComparer, ?warmup, ?verbose, ?throwOnError) =
        
        inherit PerformanceTester<'Testable>()

        do
            if otherImpls.IsEmpty then invalidArg "otherImpls" "need at least one alternative implementation."

            // check for duplicate implementations in list
            let duplicates =
                testedImpl :: otherImpls
                |> Seq.map (fun impl -> impl.Name)
                |> getDuplicates
                |> Seq.toList

            match duplicates with
            | [] -> ()
            | hd :: _ -> 
                invalidArg "otherImpls" <|
                    sprintf "Found duplicate implementation id '%s'." hd

        let comparer = match comparer with Some c -> c | None -> new TimeComparer() :> _
        let verbose = defaultArg verbose true
        let warmup = defaultArg warmup false
        let throwOnError = defaultArg throwOnError false

        let mutable thisSession = TestSession.Empty currentHost testedImpl.Name
        let otherSessions = otherImpls |> List.toArray |> Array.map (fun impl -> TestSession.Empty currentHost impl.Name)

        override __.TestedImplementation = testedImpl
        override __.RunTest (perfTest : PerfTest<'Testable>) =
            lock otherSessions (fun () ->

            let thisResult = Benchmark.Run(perfTest, testedImpl, warmup = warmup)
            thisSession <- thisSession.Append(thisResult)

            let otherResults = 
                otherImpls 
                |> List.mapi (fun i otherImpl ->
                    let r = Benchmark.Run(perfTest, otherImpl, catchExceptions = true, warmup = warmup)
                    otherSessions.[i] <- otherSessions.[i].Append(r)
                    let isFaster = comparer.IsBetterOrEquivalent thisResult r
                    let msg = comparer.GetComparisonMessage thisResult r

                    if verbose then
                        if isFaster then Console.WriteLine(msg)
                        else Console.Error.WriteLine(msg)

                    msg, r, isFaster)

            if throwOnError then
                for msg, other, isFaster in otherResults do
                    if not isFaster then
                        raise <| new PerformanceException(msg, thisResult, other))

        override __.GetTestResults () = thisSession :: List.ofArray otherSessions