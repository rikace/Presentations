using System;
using System.ComponentModel;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;
using Microsoft.Win32.SafeHandles;

namespace CommonHelpers
{   
    /*
            BenchPerformance.Time("Fibonacci", 100, () =>
            {
                range.ForEach(i => IsPrime(i));
            });*/

    public sealed class BenchPerformance : IDisposable
    {
        // Holds the stopwatch time.

        #region Delegates

 
        #endregion

        private static Int64[] m_arr;
        private static Int32 m_iterations;
        private readonly Int32 m_gen0Start;
        private readonly Int32 m_gen2Start;
        private readonly Int32 m_get1Start;
        private readonly UInt64 m_startCycles;
        private readonly Int64 m_startTime;
        private readonly String m_text;

        /// <summary>
        /// The private constructor for the class.
        /// </summary>
        /// <param name="startFresh">
        /// If true, forces a GC in order to count just new garbage collections.
        /// </param>
        /// <param name="format">
        /// A composit text string.
        /// </param>
        /// <param name="args">
        /// An array of objects to write with <paramref name="text"/>.
        /// </param>
        private BenchPerformance(Boolean startFresh,
                                 String format,
                                 params Object[] args)
        {
            if (startFresh)
            {
                PrepareForOperation();
            }

            m_text = String.Format(format, args);

            m_gen0Start = GC.CollectionCount(0);
            m_get1Start = GC.CollectionCount(1);
            m_gen2Start = GC.CollectionCount(2);

            // Get the time before returning so that any code above doesn't 
            // impact the time.
            m_startTime = Stopwatch.GetTimestamp();
            m_startCycles = CycleBench.Thread();
        }

        #region IDisposable Members

        public void Dispose()
        {
            UInt64 elapsedCycles = CycleBench.Thread() - m_startCycles;
            Int64 elapsedTime = Stopwatch.GetTimestamp() - m_startTime;
            Int64 milliseconds = (elapsedTime * 1000) / Stopwatch.Frequency;

            if (false == String.IsNullOrEmpty(m_text))
            {
                ConsoleColor defColor = Console.ForegroundColor;
                Console.ForegroundColor = ConsoleColor.Yellow;
                String title = String.Format("\tOperation > {0} <", m_text);
                String gcInfo =
                    String.Format("\tGC(G0={2,4}, G1={3,4}, G2={4,4})\n\tTotal Time  {0,7:N0}ms {1,11:N0} Kc \n",
                                  milliseconds,
                                  elapsedCycles / 1000,
                                  GC.CollectionCount(0) - m_gen0Start,
                                  GC.CollectionCount(1) - m_get1Start,
                                  GC.CollectionCount(2) - m_gen2Start);

                Console.WriteLine(new String('*', gcInfo.Length));
                Console.WriteLine();
                Console.WriteLine(title);
                Console.WriteLine();
                Console.ForegroundColor = defColor;
                if (m_arr.Length > 1)
                {
                    Console.WriteLine(String.Format("\tRepeat times {0}", m_arr.Length));
                    Console.WriteLine(String.Format("\tBest Time {0} ms", m_arr.Min()));
                    Console.WriteLine(String.Format("\tWorst Time {0} ms", m_arr.Max()));
                    Console.WriteLine(String.Format("\tAvarage Time {0} ms", m_arr.Average()));
                }
                Console.WriteLine();
                Console.WriteLine(gcInfo);
                Console.ForegroundColor = ConsoleColor.Yellow;
                Console.WriteLine(new String('*', gcInfo.Length));
                Console.ForegroundColor = ConsoleColor.Red;
                //Console.WriteLine("\t\t**** Press <ENTER> to Continue ****");
                Console.WriteLine(new String('*', gcInfo.Length));
                Console.ForegroundColor = defColor;
                //  Console.ReadKey(true);
            }
        }

        #endregion

        /// <summary>
        /// Times the operation in <paramref name="operation"/>.
        /// </summary>
        /// <param name="text">
        /// The text to display along with the timing information.
        /// </param>
        /// <param name="iterations">
        /// The number of times to execute <paramref name="operation"/>.
        /// </param>
        /// <param name="operation">
        /// The <see cref="TimedOperation"/> delegate to execute.
        /// </param>
        public static void Time(String text,
                                        Action operation,
                                        Int32 iterations = 1, Boolean startFresh = true)
        {
            m_iterations = iterations;
            using (new BenchPerformance(startFresh, text))
            {
                m_arr = new long[iterations];
                var watch = new Stopwatch();
                for (int i = 0; i < iterations; i++)
                {
                    watch.Start();
                    operation();
                    watch.Stop();
                    m_arr[i] = watch.ElapsedMilliseconds;
                    watch.Reset();
                }
            }
        }

        private static void PrepareForOperation()
        {
            // Pre-empt a lot of other apps for more accurate results
            Process.GetCurrentProcess().PriorityClass = ProcessPriorityClass.High;
            Thread.CurrentThread.Priority = ThreadPriority.Highest;
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
            // Force the BenchPerformance.Time to be jitted/loaded/whatever. This ensures 
            // that the first use does not influence the timing.
            Time(String.Empty, () => Thread.Sleep(0), 1, false);
        }
    }

    internal sealed class CycleBench
    {
        private readonly SafeWaitHandle m_handle;
        private readonly UInt64 m_startCycleTime;
        private readonly Boolean m_trackingThreadTime;

        /// <summary>
        /// Instantiates the CycleBench class.
        /// </summary>
        /// <param name="trackingThreadTime">
        /// True if you want to track the current thread time. False for 
        /// </param>
        /// <param name="handle">
        /// The handle to the process for observing.
        /// </param>
        private CycleBench(Boolean trackingThreadTime, SafeWaitHandle handle)
        {
            m_trackingThreadTime = trackingThreadTime;
            m_handle = handle;
            m_startCycleTime = m_trackingThreadTime
                                   ? Thread()
                                   : Process(m_handle);
        }

        /// <summary>
        /// Calculates the elapsed cycle times.
        /// </summary>
        /// <returns>
        /// The elapsed time.
        /// </returns>
        [CLSCompliant(false)]
        public UInt64 Elapsed()
        {
            UInt64 now = m_trackingThreadTime
                             ? Thread()
                             : Process(m_handle);
            return (now - m_startCycleTime);
        }

        /// <summary>
        /// Starts timing for the specified thread.
        /// </summary>
        /// <param name="threadHandle">
        /// The handle to the thread to time.
        /// </param>
        /// <returns>
        /// A <see cref="CycleBench"/> instance for timing the specified thread.
        /// </returns>
        public static CycleBench StartThread(SafeWaitHandle threadHandle)
        {
            return new CycleBench(true, threadHandle);
        }

        /// <summary>
        /// Stars timing for the specified process.
        /// </summary>
        /// <param name="processHandle">
        /// The handle to the process to time.
        /// </param>
        /// <returns>
        /// A <see cref="CycleBench"/> instance for timing the specified process.
        /// </returns>
        public static CycleBench StartProcess(SafeWaitHandle processHandle)
        {
            return new CycleBench(false, processHandle);
        }

        /// <summary>
        /// Retrieves the cycle time for the specified thread.
        /// </summary>
        /// <param name="threadHandle">
        /// Identifies the thread whose cycle time you'd like to obtain.
        /// </param>
        /// <returns>
        /// The thread's cycle time.
        /// </returns>
        /// <exception cref="Win32Exception">
        /// Thrown if the underlying timing API fails.
        /// </exception>
        [CLSCompliant(false)]
        public static UInt64 Thread(SafeWaitHandle threadHandle)
        {
            UInt64 cycleTime;
            if (!QueryThreadCycleTime(threadHandle, out cycleTime))
            {
                throw new Win32Exception();
            }
            return cycleTime;
        }

        /// <summary>
        /// Retrieves the cycle time for the current thread.
        /// </summary>
        /// <returns>
        /// The thread's cycle time.
        /// </returns>
        /// <exception cref="Win32Exception">
        /// Thrown if the underlying timing API fails.
        /// </exception>
        [CLSCompliant(false)]
        public static UInt64 Thread()
        {
            UInt64 cycleTime;
            if (!QueryThreadCycleTime((IntPtr)(-2), out cycleTime))
            {
                throw new Win32Exception();
            }
            return cycleTime;
        }

        /// <summary>
        /// Retrieves the sum of the cycle time of all threads of the specified 
        /// process.
        /// </summary>
        /// <param name="processHandle">
        /// Identifies the process whose threads' cycles times you'd like to 
        /// obtain.
        /// </param>
        /// <returns>
        /// The process' cycle time.
        /// </returns>
        /// <exception cref="Win32Exception">
        /// Thrown if the underlying timing API fails.
        /// </exception>
        [CLSCompliant(false)]
        public static UInt64 Process(SafeWaitHandle processHandle)
        {
            UInt64 cycleTime;
            if (!QueryProcessCycleTime(processHandle, out cycleTime))
            {
                throw new Win32Exception();
            }
            return cycleTime;
        }

        /// <summary>
        /// Retrieves the cycle time for the idle thread of each processor in 
        /// the system.
        /// </summary>
        /// <returns>
        /// The number of CPU clock cycles used by each idle thread.
        /// </returns>
        /// <exception cref="Win32Exception">
        /// Thrown if the underlying timing API fails.
        /// </exception>
        [CLSCompliant(false)]
        public static UInt64[] IdleProcessors()
        {
            Int32 byteCount = Environment.ProcessorCount;
            var cycleTimes = new UInt64[byteCount];
            byteCount *= 8; // Size of UInt64
            if (!QueryIdleProcessorCycleTime(ref byteCount, cycleTimes))
            {
                throw new Win32Exception();
            }
            return cycleTimes;
        }

        [DllImport("Kernel32", ExactSpelling = true, SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern Boolean
            QueryThreadCycleTime(IntPtr threadHandle, out UInt64 CycleTime);


        [DllImport("Kernel32", ExactSpelling = true, SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern Boolean
            QueryThreadCycleTime(SafeWaitHandle threadHandle,
                                 out UInt64 CycleTime);

        [DllImport("Kernel32", ExactSpelling = true, SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern Boolean
            QueryProcessCycleTime(SafeWaitHandle processHandle,
                                  out UInt64 CycleTime);

        [DllImport("Kernel32", ExactSpelling = true, SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern Boolean
            QueryIdleProcessorCycleTime(ref Int32 byteCount,
                                        UInt64[] CycleTimes);
    }
}