namespace PerfUtil

    open System
    open System.Collections.Generic

    open PerfUtil.Utils
    open PerfUtil.Persist

    /// <summary>Compares current implementation against a collection of past tests.</summary>
    /// <param name="currentImpl">Implementation under test.</param>
    /// <param name="testRunId">Unique identifier of current implementation.</param>
    /// <param name="historyFile">Specifies path to persisted past test results. Defaults to 'PerfUtil.DefaultPersistenceFile'.</param>
    /// <param name="verbose">Print performance results to stdout.</param>
    /// <param name="warmup">Perform a warmup run before attempting benchmark. Defaults to false.</param>
    /// <param name="throwOnError">Raise an exception if performance comparison fails. Defaults to false.</param>
    /// <param name="overwrite">Overwrite sessions with identical run id, if such session exists. Defaults to true.</param>
    type PastImplementationComparer<'Testable when 'Testable :> ITestable>
        (currentImpl : 'Testable, testRunId : string, ?historyFile : string, ?warmup,
            ?comparer : IPerformanceComparer, ?verbose : bool, ?throwOnError : bool, ?overwrite : bool) =

        inherit PerformanceTester<'Testable> ()

        let comparer = match comparer with Some p -> p | None -> new TimeComparer() :> _ 
        let verbose = defaultArg verbose true
        let throwOnError = defaultArg throwOnError false
        let overwrite = defaultArg overwrite true
        let warmup = defaultArg warmup false
        let historyFile = defaultArg historyFile PerfUtil.DefaultPersistenceFile

        let mutable currentSession = TestSession.Empty currentHost testRunId
        let pastSessions = 
            match sessionsOfFile historyFile with
            | Some(id, sessions) when id = currentImpl.Name -> sessions
            | Some(id,_) -> 
                invalidOp <| 
                    sprintf "PerfUtil: Expected session id '%s', but '%s' contains id '%s'."
                        currentImpl.Name historyFile id
            | None -> []

        let isCommited = ref false

        do
            if pastSessions |> List.exists (fun s -> s.Id = testRunId) then
                let msg = sprintf "a past test with id '%s' already exists in history file." testRunId
                if not overwrite then invalidOp msg
                elif verbose then
                    Console.Error.WriteLine(sprintf "WARNING: %s" msg)

            match pastSessions |> List.tryFind (fun s -> s.Hostname <> currentHost) with
            | Some session ->
                if verbose then
                    let msg = 
                        sprintf "WARNING: Past session '%s' was performed on foreign host '%s'." 
                            session.Id session.Hostname
                    Console.Error.WriteLine(msg)
            | _ -> ()

            let duplicates =
                pastSessions
                |> Seq.map (fun s -> s.Id)
                |> getDuplicates
                |> Seq.toList

            match duplicates with
            | [] -> ()
            | hd :: _ -> 
                invalidOp <| sprintf "Found duplicate implementation id '%s'." hd

        let compareResultWithHistory (current : PerfResult) =
            let olderRuns =
                pastSessions
                |> List.choose (fun s -> s.Results.TryFind current.TestId)
                |> List.map (fun older -> 
                    let isFaster = comparer.IsBetterOrEquivalent current older
                    let msg = comparer.GetComparisonMessage current older
                    older, isFaster, msg)

            if verbose then
                for _, isFaster, msg in olderRuns do
                    if isFaster then Console.WriteLine(msg)
                    else Console.Error.WriteLine(msg)

            if throwOnError then
                for older, isFaster, msg in olderRuns do
                    if not isFaster then 
                        raise <| new PerformanceException(msg, current, older)

        /// <summary>Compares current implementation against a collection of past tests.</summary>
        /// <param name="currentImpl">Implementation under test.</param>
        /// <param name="version">Version number of current implementation.</param>
        /// <param name="warmup">Perform a warmup run before attempting benchmark. Defaults to false.</param>
        /// <param name="historyFile">Specifies path to persisted past test results. Defaults to 'PerfUtil.DefaultPersistenceFile'.</param>
        /// <param name="verbose">Print performance results to stdout.</param>
        /// <param name="throwOnError">Raise an exception if performance comparison fails. Defaults to false.</param>
        /// <param name="overwrite">Overwrite sessions with identical run id, if such session exists. Defaults to false.</param>
        new (currentImpl : 'Testable, version : Version, ?historyFile : string, ?warmup,
                ?comparer : IPerformanceComparer, ?verbose : bool, ?throwOnError : bool, ?overwrite : bool) =

            new PastImplementationComparer<'Testable>
                (currentImpl, sprintf "%s v.%O" currentImpl.Name version, ?historyFile = historyFile, ?warmup = warmup,
                    ?comparer = comparer, ?verbose = verbose, ?throwOnError = throwOnError, ?overwrite = overwrite)

        override __.TestedImplementation = currentImpl

        override __.RunTest (perfTest : PerfTest<'Testable>) =
            if isCommited.Value then invalidOp "Test run has been finalized."
            lock currentSession (fun () ->
                let result = Benchmark.Run(perfTest, currentImpl, warmup = warmup, sessionId = testRunId, testId = perfTest.Id)
                currentSession <- currentSession.Append(result)
                do compareResultWithHistory result)

        override __.GetTestResults () = currentSession :: pastSessions

        /// <summary>append current test results to persistence file.</summary>
        /// <param name="file">Optionally, specifies a file path to persist to.</param>
        member __.PersistCurrentResults (?file) =
            let file = defaultArg file historyFile
            lock isCommited (fun () ->
                match isCommited.Value with
                | true -> invalidOp "Cannot commit results twice."
                | false ->
                    let pastSessions = pastSessions |> List.filter (fun s -> s.Id <> testRunId)
                    sessionsToFile currentImpl.Name file (currentSession :: pastSessions)
                    isCommited := true)