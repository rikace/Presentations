namespace PerfUtil

    open System
    open System.Reflection

    open PerfUtil.Utils

    // benchmarking code, taken from FSI timer implementation

    type Benchmark private () =
            
        static let lockObj = box 42
        static let proc = System.Diagnostics.Process.GetCurrentProcess()
        static let numGC = System.GC.MaxGeneration

        /// <summary>Benchmarks a given computation.</summary>
        /// <param name="testF">Test function.</param>
        /// <param name="state">Input state to the test function.</param>
        /// <param name="repeat">Number of times to repeat the benchmark. Defaults to 1.</param>
        /// <param name="warmup">Perform a warmup run before attempting benchmark. Defaults to false.</param>
        /// <param name="sessionId">Test session identifier given to benchmark. Defaults to empty string.</param>
        /// <param name="testId">Test identifier given to benchmark. Defaults to empty string.</param>
        /// <param name="catchExceptions">Catches exceptions raised by the test function. Defaults to false.</param>
        static member Run<'State>(testF : 'State -> unit, state : 'State, ?repeat, ?warmup, ?sessionId, ?testId, ?catchExceptions) =
            let repeat = defaultArg repeat 1
            let catchExceptions = defaultArg catchExceptions false
            let warmup = defaultArg warmup false
            let testId = defaultArg testId ""
            let sessionId = defaultArg sessionId ""

            lock lockObj (fun () ->

            let stopwatch = new System.Diagnostics.Stopwatch()

            if warmup then
                try testF state
                with e when catchExceptions -> ()

            do 
                GC.Collect(3)
                GC.WaitForPendingFinalizers()
                GC.Collect(3)
                System.Threading.Thread.Sleep(100)


            let gcDelta = Array.zeroCreate<int> (numGC + 1)
            let inline computeGcDelta () =
                for i = 0 to numGC do
                    gcDelta.[i] <- System.GC.CollectionCount(i) - gcDelta.[i]

            do computeGcDelta ()
            let startTotal = proc.TotalProcessorTime
            let date = DateTime.Now
            stopwatch.Start()

            let error = 
                try 
                    for i = 1 to repeat do testF state 
                    None 
                with e when catchExceptions -> Some e.Message

            stopwatch.Stop()
            let total = proc.TotalProcessorTime - startTotal
            do computeGcDelta ()

            {
                Date = date
                TestId = testId
                SessionId = sessionId

                Error = error

                Repeat = repeat

                Elapsed = stopwatch.Elapsed
                CpuTime = total
                GcDelta = Array.toList gcDelta
            })
            

        /// <summary>Benchmarks a given computation.</summary>
        /// <param name="testF">Test function.</param>
        /// <param name="repeat">Number of times to repeat the benchmark. Defaults to 1.</param>
        /// <param name="warmup">Perform a warmup run before attempting benchmark. Defaults to false.</param>
        /// <param name="sessionId">Test session identifier given to benchmark. Defaults to empty string.</param>
        /// <param name="testId">Test identifier given to benchmark. Defaults to empty string.</param>
        /// <param name="catchExceptions">Catches exceptions raised by the test function. Defaults to false.</param>
        static member Run(testF : unit -> unit, ?repeat, ?warmup, ?sessionId, ?testId, ?catchExceptions) =
            Benchmark.Run(testF, (), ?repeat = repeat, ?sessionId = sessionId, ?warmup = warmup,
                                        ?testId = testId, ?catchExceptions = catchExceptions)

        /// <summary>Runs a given performance test.</summary>
        /// <param name="testF">Performance test.</param>
        /// <param name="impl">Implementation to run the performance test on.</param>
        /// <param name="warmup">Perform a warmup run before attempting benchmark. Defaults to false.</param>
        /// <param name="catchExceptions">Catches exceptions raised by the test function. Defaults to false.</param>
        /// <param name="sessionId">Test session identifier given to benchmark. Defaults to empty string.</param>
        /// <param name="testId">Test identifier given to benchmark. Defaults to empty string.</param>
        static member Run(perfTest : PerfTest<'Impl>, impl : 'Impl, ?warmup, ?catchExceptions, ?sessionId, ?testId) =
            try 
                do impl.Init()
                let testId = defaultArg testId perfTest.Id
                let sessionId = defaultArg sessionId impl.Name
                Benchmark.Run(perfTest.Test, impl, sessionId = sessionId, testId = testId, ?warmup = warmup,
                                                repeat = perfTest.Repeat, ?catchExceptions = catchExceptions)
            finally impl.Fini()