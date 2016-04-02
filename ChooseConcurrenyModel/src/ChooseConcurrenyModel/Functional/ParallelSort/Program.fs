open System
open System.Globalization
open Utilities 
open ParallelSort

/// Create array of given length and populate with random integers 
let makeArray length seed = 
    let max = 1000000
    let r = new Random(seed) 
    Array.init length (fun _ -> r.Next(max))


/// Print the first and last few elements in given array
let printElements count (array:_[]) = 
    printf "["
    for i in 0 .. count/2 do printf "%d " array.[i]
    printf "... "
    for i in 0 .. count/2 do printf "%d " array.[array.Length - 1 - count/2 + i]
    printfn "], %d elements" array.Length

// ------------------------------------------------------------------------------
// Command line arguments are:
//   length - of array to sort
//   threshold -  array length to use InsertionSort instead of SequentialQuickSort
// ------------------------------------------------------------------------------

[<EntryPoint>]
let main (args:string[]) =
    Console.WriteLine("Sort Sample\n")
#if DEBUG
    Console.WriteLine("For most accurate timing results, use Release build.\n")
#endif
    let seed = 1 
    let length =40000000

    // Generate array & run the sorting sequentially
    let a = makeArray length seed
    printElements 8 a
    SampleUtilities.TimedRun  "  Sequential QuickSort" (fun () -> 
        Sort.sequentialQuickSort a
        a.Length ) |> ignore
    printElements 8 a
    Console.WriteLine()

    // Generate array & run the sorting in parallel
    let a = makeArray length seed
    printElements 8 a 
    SampleUtilities.TimedRun "      Parallel QuickSort" (fun () -> 
        Sort.parallelQuickSort a
        a.Length ) |> ignore
    printElements 8 a 

    Console.WriteLine("\nRun complete... press enter to finish.") 
    Console.ReadKey() |> ignore
    0
