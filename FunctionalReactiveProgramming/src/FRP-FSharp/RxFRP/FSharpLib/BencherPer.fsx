
open System
open System.Diagnostics

module InteropWithNative =
    [<System.Runtime.InteropServices.DllImportAttribute("psapi.dll")>]
    extern int EmptyWorkingSet (System.IntPtr hProcess)


module Measure =
    
    type System.Int64 with 
        static member DivideByInt (n: System.Int64) (d: int) = n / (int64 d)
    
    [<System.Runtime.InteropServices.DllImportAttribute("psapi.dll")>]
    let EmptyWorkingSet() =
        let hThisProcess = Process.GetCurrentProcess().Handle
        InteropWithNative.EmptyWorkingSet(hThisProcess)
        
    let forceWorkingSet() =
        EmptyWorkingSet() |> ignore
        GC.Collect()
        GC.WaitForPendingFinalizers()
        GC.Collect()
    
    let start (startFresh:bool) (iteration:int) (text:string)  f =
        let m_gen0Start = GC.CollectionCount(0)
        let m_get1Start = GC.CollectionCount(1)
        let m_gen2Start = GC.CollectionCount(2)
        let m_startTime = Stopwatch.GetTimestamp()

        if startFresh then forceWorkingSet()

        let watch = new Stopwatch()
        let rec performAction itr acc =
            match itr with 
            | 0 ->  acc
            | _ ->  watch.Start()
                    f()
                    watch.Stop()
                    let time = watch.ElapsedMilliseconds
                    performAction (itr - 1) (time::acc)
        
        let arrResults = performAction iteration []

        { new IDisposable with
            member x.Dispose() =
                
                let elapsedTime = Stopwatch.GetTimestamp() - m_startTime;
                let milliseconds = (elapsedTime*int64(1000))/Stopwatch.Frequency;

                let defColor = Console.ForegroundColor
                Console.ForegroundColor <- ConsoleColor.Yellow
                let title = String.Format("\tOperation > {0} <", text)
                let gcInfo =
                    String.Format("\tGC(G0={2,4}, G1={3,4}, G2={4,4})\n\tTotal Time  {0,7:N0}ms",
                                  milliseconds,                                  
                                  GC.CollectionCount(0) - m_gen0Start,
                                  GC.CollectionCount(1) - m_get1Start,
                                  GC.CollectionCount(2) - m_gen2Start)

                Console.WriteLine(new String('*', gcInfo.Length))
                Console.WriteLine()
                Console.WriteLine(title)
                Console.WriteLine()
                Console.ForegroundColor <- defColor
                if List.length arrResults > 1 then
                    Console.WriteLine(String.Format("\tRepeat times {0}", (List.length arrResults)))
                    Console.WriteLine(String.Format("\tBest Time {0} ms",(List.min arrResults)))
                    Console.WriteLine(String.Format("\tWorst Time {0} ms", (List.max arrResults)))
                    Console.WriteLine(String.Format("\tAvarage Time {0} ms", (arrResults |> List.map float |>  List.average)))
                
                Console.WriteLine()
                Console.WriteLine(gcInfo)
                Console.ForegroundColor <- ConsoleColor.Yellow;
                Console.WriteLine(new String('*', gcInfo.Length))
                Console.ForegroundColor <- ConsoleColor.Red
                Console.WriteLine("\t\t**** Press <ENTER> to Continue ****")
                Console.ForegroundColor <- defColor
            }

    let startFresh = start true

