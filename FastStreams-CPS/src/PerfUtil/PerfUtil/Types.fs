namespace PerfUtil

    open System
    open System.IO
    open System.Text

    /// An abstract implementation interface
    type ITestable =
        /// Implementation name.
        abstract Name : string

        /// Run before each test run
        abstract Init : unit -> unit
        /// Run after each test run
        abstract Fini : unit -> unit

    /// Represents a performance test for a given class of implementations.
    type PerfTest<'Testable when 'Testable :> ITestable> =
        {
            Id : string
            Repeat : int
            Test : 'Testable -> unit
        }

    type PerfTest =
        /// <summary>
        ///     Defines a new PerfTest instance
        /// </summary>
        /// <param name="testF">The test function.</param>
        /// <param name="id">Test id.</param>
        /// <param name="repeat">Number of repetitions.</param>
        static member Create<'Testable when 'Testable :> ITestable>(testF, ?id, ?repeat) : PerfTest<'Testable> =
            {
                Test = testF
                Id = match id with Some i -> i | None -> testF.GetType().Name
                Repeat = defaultArg repeat 1
            }

    /// abstract performance tester
    [<AbstractClass>]
    type PerformanceTester<'Testable when 'Testable :> ITestable> () =

        /// The implementation under test.
        abstract TestedImplementation : 'Testable
        /// Run a performance test.
        abstract RunTest : PerfTest<'Testable> -> unit
        /// Get accumulated test results.
        abstract GetTestResults : unit -> TestSession list

        /// <summary>
        ///   Benchmarks given function.  
        /// </summary>
        /// <param name="testF">The test function.</param>
        /// <param name="id">Test id.</param>
        /// <param name="repeat">Number of repetitions.</param>
        member __.Run (testF : 'Testable -> unit, ?id, ?repeat) = 
            let test = PerfTest.Create(testF, ?id = id, ?repeat = repeat)
            __.RunTest test

    /// compares between two performance results
    and IPerformanceComparer =
        /// Decides if current performance is better or equivalent to the other/older performance.
        abstract IsBetterOrEquivalent : current:PerfResult -> other:PerfResult -> bool
        /// Returns a message based on comparison of the two benchmarks.
        abstract GetComparisonMessage : current:PerfResult -> other:PerfResult -> string

    /// Represents a collection of tests performed in a given run.
    and TestSession =
        {   
            Id : string
            Date : DateTime
            /// host id that performed given test.
            Hostname : string
            /// results indexed by test id
            Results : Map<string, PerfResult>
        }
    with
        member internal s.Append(br : PerfResult, ?overwrite) =
            let overwrite = defaultArg overwrite true
            if not overwrite && s.Results.ContainsKey br.TestId then
                invalidOp <| sprintf "A test '%s' has already been recorded." br.TestId

            { s with Results = s.Results.Add(br.TestId, br) }

        static member internal Empty hostname (id : string) =
            {
                Id = id
                Hostname = hostname
                Date = DateTime.Now
                Results = Map.empty
            }

    /// Contains performance information
    and PerfResult =
        {
            /// Test identifier
            TestId : string
            /// Test session identifier
            SessionId : string
            /// Execution date
            Date : DateTime

            /// Catch potential error message
            Error : string option

            /// Number of times the test was run
            Repeat : int

            Elapsed : TimeSpan
            CpuTime : TimeSpan
            /// Garbage collect differential per generation
            GcDelta : int list
        }
    with
        override r.ToString () =
            let sb = new StringBuilder()
            sb.Append(sprintf "%s: Real: %O, CPU: %O" r.TestId r.Elapsed r.CpuTime) |> ignore
            r.GcDelta |> List.iteri (fun g i -> sb.Append(sprintf ", gen%d: %d" g i) |> ignore)
            sb.Append(sprintf ", Date: %O" r.Date) |> ignore
            sb.ToString()

        member r.HasFailed = r.Error.IsSome


    type PerformanceException (message : string, this : PerfResult, other : PerfResult) =
        inherit System.Exception(message)

        do assert(this.TestId = other.TestId)

        member __.TestId = this.TestId
        member __.CurrentTestResult = this
        member __.OtherTestResult = other

    /// indicates that given method is a performance test
    type PerfTestAttribute(repeat : int) =
        inherit System.Attribute()
        new () = new PerfTestAttribute(1)
        member __.Repeat = repeat

    type PerfUtil private () =
        static let mutable result = 
            let libPath = 
                System.Reflection.Assembly.GetExecutingAssembly().Location 
                |> Path.GetDirectoryName

            Path.Combine(libPath, "perfResults.xml")

        /// gets or sets the default persistence file used by the PastSessionComparer
        static member DefaultPersistenceFile
            with get () = result
            and set path =
                if not <| File.Exists(Path.GetDirectoryName path) then
                    invalidOp <| sprintf "'%s' is not a valid path." path

                lock result (fun () -> result <- path)