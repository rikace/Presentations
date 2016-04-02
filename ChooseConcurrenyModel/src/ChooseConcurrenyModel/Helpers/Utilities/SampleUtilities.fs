
namespace Utilities

open System
open System.Collections.Generic
open System.Diagnostics
open System.Globalization
open System.IO
open System.Linq
open System.Threading

/// Static class that contains timing and numerical utilities
module SampleUtilities = 

    // --------------------------------------------------------------------------
    // Timing utilities
    // --------------------------------------------------------------------------

    /// <summary>
    /// TimeSpan pretty printer
    /// </summary>
    /// <param name="ts">The TimeSpan to format</param>
    /// <returns>A formatted string</returns>
    let FormattedTime(ts:TimeSpan) =
         String.Format
          ( "{0:00}:{1:00}:{2:00}.{3:00}",
            ts.Hours, ts.Minutes, ts.Seconds,
            ts.Milliseconds / 10 )

    /// Format and print elapsed time returned by Stopwatch
    let PrintTime(ts:TimeSpan) = 
        Console.WriteLine(FormattedTime(ts))

    /// Executes a function and prints timing results
    let TimedAction (label:string) test =
        Console.WriteLine("Starting {0}", label)
        let stopWatch = new Stopwatch()
        stopWatch.Start()

        let result = test()

        stopWatch.Stop()
        let seqT = stopWatch.Elapsed
        Console.WriteLine("{0}: {1}", label, FormattedTime(seqT))
        Console.WriteLine()
        result

    /// Executes a function and prints timing results
    let TimedRun (label:string) test = 
        let stopWatch = new Stopwatch()
        stopWatch.Start()

        let result = test()

        stopWatch.Stop()
        let seqT = stopWatch.Elapsed
        Console.WriteLine("{0} (result={1}): {2}", label, result.ToString(), FormattedTime(seqT))
        result

    /// <summary>
    /// Simulates a CPU-intensive operation on a single core. The operation will use approximately 100% of a
    /// single CPU for a specified duration.
    /// </summary>
    /// <param name="seconds">The approximate duration of the operation in seconds</param>
    /// <param name="token">A token that may signal a request to cancel the operation.</param>
    /// <param name="throwOnCancel">true if an execption should be thrown in response to a cancellation request.</param>
    /// <returns>true if operation completed normally false if the user canceled the operation</returns>
    let DoCpuIntensiveOperation seconds (token:CancellationToken) throwOnCancel =
        if (token.IsCancellationRequested) then
            if (throwOnCancel) then token.ThrowIfCancellationRequested()
            false
        else
            let ms = int64 (seconds * 1000.0)
            let sw = new Stopwatch()
            sw.Start()
            let checkInterval = Math.Min(20000000, int (20000000.0 * seconds))

            // Loop to simulate a computationally intensive operation
            let rec loop i = 
                // Periodically check to see if the user has requested 
                // cancellation or if the time limit has passed
                let check = seconds = 0.0 || i % checkInterval = 0
                if check && token.IsCancellationRequested then
                    if throwOnCancel then token.ThrowIfCancellationRequested()
                    false
                elif check && sw.ElapsedMilliseconds > ms then
                    true
                else 
                  loop (i + 1)
          
            // Start the loop with 0 as the first value
            loop 0

    /// <summary>
    /// Simulates a CPU-intensive operation on a single core. The operation will use approximately 100% of a
    /// single CPU for a specified duration.
    /// </summary>
    /// <param name="seconds">The approximate duration of the operation in seconds</param>
    /// <returns>true if operation completed normally false if the user canceled the operation</returns>
    let DoCpuIntensiveOperationSimple seconds =
        DoCpuIntensiveOperation seconds CancellationToken.None false


    // vary to simulate I/O jitter
    let SleepTimeouts = 
      [| 65; 165; 110; 110; 185; 160; 40; 125; 275; 110; 80; 190; 70; 165; 
         80; 50; 45; 155; 100; 215; 85; 115; 180; 195; 135; 265; 120; 60; 
         130; 115; 200; 105; 310; 100; 100; 135; 140; 235; 205; 10; 95; 175; 
         170; 90; 145; 230; 365; 340; 160; 190; 95; 125; 240; 145; 75; 105; 
         155; 125; 70; 325; 300; 175; 155; 185; 255; 210; 130; 120; 55; 225;
         120; 65; 400; 290; 205; 90; 250; 245; 145; 85; 140; 195; 215; 220;
         130; 60; 140; 150; 90; 35; 230; 180; 200; 165; 170; 75; 280; 150; 
         260; 105 |]

    /// <summary>
    /// Simulates an I/O-intensive operation on a single core. The operation will use only a small percent of a
    /// single CPU's cycles however, it will block for the specified number of seconds.
    /// </summary>
    /// <param name="seconds">The approximate duration of the operation in seconds</param>
    /// <param name="token">A token that may signal a request to cancel the operation.</param>
    /// <param name="throwOnCancel">true if an execption should be thrown in response to a cancellation request.</param>
    /// <returns>true if operation completed normally false if the user canceled the operation</returns>
    let DoIoIntensiveOperation seconds (token:CancellationToken) throwOnCancel =
        if token.IsCancellationRequested then false else
        let ms = int (seconds * 1000.0)
        let sw = new Stopwatch()
        let timeoutCount = SleepTimeouts.Length
        sw.Start()

        // Loop to simulate I/O intensive operation
        let mutable i = Math.Abs(sw.GetHashCode()) % timeoutCount
        let mutable result = None
        while result = None do
            let timeout = SleepTimeouts.[i]
            i <- (i + 1) % timeoutCount

            // Simulate I/O latency
            Thread.Sleep(timeout)

            // Has the user requested cancellation? 
            if token.IsCancellationRequested then
                if throwOnCancel then token.ThrowIfCancellationRequested()
                result <- Some false

            // Is the computation finished?
            if sw.ElapsedMilliseconds > int64 ms then
                result <- Some true
      
        result.Value


    /// Simulates an I/O-intensive operation on a single core. The operation will 
    /// use only a small percent of a single CPU's cycles however, it will block 
    /// for the specified number of seconds.
    ///
    /// This is same as 'DoIoIntensiveOperation', but uses F# asyncs to simulate
    /// non-blocking (asynchronous) I/O typical in F# async applications.
    let AsyncDoIoIntensiveOperation seconds (token:CancellationToken) throwOnCancel = 
      async { if token.IsCancellationRequested then return false else
              let ms = int (seconds * 1000.0)
              let sw = new Stopwatch()
              let timeoutCount = SleepTimeouts.Length
              sw.Start()

              // Loop to simulate I/O intensive operation
              let i = ref (Math.Abs(sw.GetHashCode()) % timeoutCount)
              let result = ref None
              while !result = None do
                  let timeout = SleepTimeouts.[!i]
                  i := (!i + 1) % timeoutCount

                  // Simulate I/O latency
                  do! Async.Sleep(timeout)

                  // Has the user requested cancellation? 
                  if token.IsCancellationRequested then
                      if throwOnCancel then token.ThrowIfCancellationRequested()
                      result := Some false

                  // Is the computation finished?
                  if sw.ElapsedMilliseconds > int64 ms then
                      result := Some true
      
              return result.Value.Value }

    // --------------------------------------------------------------------------
    // File utilities
    // --------------------------------------------------------------------------
  

    /// Check whether directory exists, if not write message and exit immediately.
    let CheckDirectoryExists dirName =
        if not (Directory.Exists(dirName)) then
            Console.WriteLine("Directory does not exist: {0}", dirName)
            Environment.Exit(0)

    /// Check whether file exists, if not write message and exit immediately.
    /// (can't use this method to check whether directory exists)
    let CheckFileExists path =
        if not (File.Exists(path)) then
            Console.WriteLine("File does not exist: {0}", path)
            Environment.Exit(0)

    /// <summary>
    /// Get names of image files in directory
    /// </summary>
    /// <param name="sourceDir">Name of directory</param>
    /// <param name="maxImages">Maximum number of image file names to return</param>
    /// <returns>List of image file names in directory (basenames not including directory path)</returns>
    let GetImageFilenamesList sourceDir maxImages =
        let fileNames = new ResizeArray<string>()
        let dirInfo = new DirectoryInfo(sourceDir)
        let files = 
          dirInfo.GetFiles()
          |> Array.filter (fun file ->
                // LIMITATION - only handles jpg, not gif, png etc.
                file.Extension.ToUpper(CultureInfo.InvariantCulture) = ".JPG")
          |> Array.map (fun file -> file.Name)

        let imageFileCount = min maxImages files.Length
        files 
          |> Seq.take imageFileCount
          |> Seq.sort |> Seq.toList


    /// Repeatedly loop through all of the files in the source directory. This
    /// enumerable has an infinite number of values.
    let GetImageFilenames sourceDir maxImages = 
      seq { let names = GetImageFilenamesList sourceDir maxImages
            while true do
                yield! names }

    // --------------------------------------------------------------------------
    // Numerical Routines
    // --------------------------------------------------------------------------

    /// Return array of floats for indices 0 .. count-1
    let Range count = 
        if count < 0 then
            raise (new ArgumentOutOfRangeException("count"))
        Array.init count float


    /// <summary>
    /// Linear regression of (x, y) pairs
    /// </summary>
    /// <param name="abscissaValues">The x values</param>
    /// <param name="ordinateValues">The y values corresponding to each x value</param>
    /// <returns>A trend line that best predicts each (x, y) pair</returns>
    let Fit (abscissaValues:float[]) (ordinateValues:float[]) =
        if abscissaValues = null then nullArg "abscissaValues"
        if ordinateValues = null then nullArg "ordinateValues"
        if abscissaValues.Length <> ordinateValues.Length then
            invalidArg "abscissaValues and ordinateValues" "must contain the same number of values"
        if abscissaValues.Length < 2 then
            invalidArg "abscissaValues" "must contain at least two elements"

        let abscissaMean = abscissaValues.Average()
        let ordinateMean = ordinateValues.Average()

        // Calculate the sum of squared differences
        let mutable xx = 0.0
        let mutable xy = 0.0
        for i in 0 .. abscissaValues.Length - 1 do
            let abs = abscissaValues.[i]
            let ord = ordinateValues.[i] 
            let xi = abs - abscissaMean
            xx <- xx + xi * xi
            xy <- xy + xi * (ord - ordinateMean)

        if xx = 0.0 then
           raise (new ArgumentException("abscissaValues must not all be coincident"))
        let slope = xy / xx
        Trend(slope, ordinateMean - slope * abscissaMean)


    /// <summary>
    /// Linear regression with x-values given implicity by the y-value indices
    /// </summary>
    /// <param name="ordinateValues">A series of two or more values</param>
    /// <returns>A trend line</returns>
    let FitImplicit (ordinateValues:_[]) =
        if ordinateValues = null then
            raise (new ArgumentNullException("ordinateValues"))
        // Special case - x values are just the indices of the y's
        Fit (Range(ordinateValues.Length)) ordinateValues


    /// Adaptation of Peter J. Acklam's Perl implementation. 
    /// See http://home.online.no/~pjacklam/notes/invnorm/
    /// This approximation has a relative error of 1.15 × 10−9 or less. 
    let GaussianInverseSimple value =
  
        // Lower and upper breakpoints
        let plow = 0.02425
        let phigh = 1.0 - plow

        let p = if (phigh < value) then 1.0 - value else value
        let sign = if (phigh < value) then -1.0 else 1.0

        if p < plow then
            // Rational approximation for tail
            let c = [| -7.784894002430293e-03; -3.223964580411365e-01;
                       -2.400758277161838e+00; -2.549732539343734e+00;
                       4.374664141464968e+00; 2.938163982698783e+00 |]
            let d = [| 7.784695709041462e-03; 3.224671290700398e-01;
                       2.445134137142996e+00; 3.754408661907416e+00 |]
            let q = Math.Sqrt(-2.0 * Math.Log(p))
            sign * (((((c.[0] * q + c.[1]) * q + c.[2]) * q + c.[3]) * q + c.[4]) * q + c.[5]) /
                                     ((((d.[0] * q + d.[1]) * q + d.[2]) * q + d.[3]) * q + 1.0)

        else
            // Rational approximation for central region
            let a = [| -3.969683028665376e+01; 2.209460984245205e+02;
                       -2.759285104469687e+02; 1.383577518672690e+02;
                       -3.066479806614716e+01; 2.506628277459239e+00 |]
            let b = [| -5.447609879822406e+01; 1.615858368580409e+02;
                       -1.556989798598866e+02; 6.680131188771972e+01;
                       -1.328068155288572e+01 |]
            let q = p - 0.5
            let r = q * q
            (((((a.[0] * r + a.[1]) * r + a.[2]) * r + a.[3]) * r + a.[4]) * r + a.[5]) * q /
                                      (((((b.[0] * r + b.[1]) * r + b.[2]) * r + b.[3]) * r + b.[4]) * r + 1.0)


    /// <summary>
    /// Calculates an approximation of the inverse of the cumulative normal distribution.
    /// </summary>
    /// <param name="cumulativeDistribution">The percentile as a fraction (.50 is the fiftieth percentile). 
    /// Must be greater than 0 and less than 1.</param>
    /// <param name="mean">The underlying distribution's average (i.e., the value at the 50th percentile) (</param>
    /// <param name="standardDeviation">The distribution's standard deviation</param>
    /// <returns>The value whose cumulative normal distribution (given mean and stddev) is the percentile given as an argument.</returns>
    let GaussianInverse cumulativeDistribution mean standardDeviation = 
        if not (0.0 < cumulativeDistribution && cumulativeDistribution < 1.0) then
            raise (new ArgumentOutOfRangeException("cumulativeDistribution"))

        let result = GaussianInverseSimple cumulativeDistribution
        mean + result * standardDeviation

  
    // ----------------------------------------------------------------------------
    // Other Utilities
    // ----------------------------------------------------------------------------
       
    /// <summary>
    /// Creates a seed that does not depend on the system clock. A 
    /// unique value will be created with each invocation.
    /// </summary>
    /// <returns>An integer that can be used to seed a random generator</returns>
    /// <remarks>This method is thread safe.</remarks>
    let MakeRandomSeed() = 
        Guid.NewGuid().ToString().GetHashCode()