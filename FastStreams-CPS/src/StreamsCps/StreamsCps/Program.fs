open System
open System.Reactive.Linq
open SimpleStreams
open Stream

// For simple pipelines we have observed performance improvements of a 
// factor of four and for more complex pipelines the performance gains 
// are even greater. Important performance tip: Make sure that FSI is 
// running with 64-bit option set to true and fsproj option 
// prefer 32-bit is unchecked.

[<EntryPoint>]
let main argv = 

    let dataMedium = [| 1..10000000 |] |> Array.map (fun x -> x % 1000) |> Array.map int64
    let dataHigh = [| 1..1000000 |] |> Array.map int64
    let dataLow = [| 1..10 |] |> Array.map int64

    let data = dataHigh

    let sumSeq() = Seq.sum data
    let sumArray () = Array.sum data
    let sumStreams () = Stream.ofArray data |> Stream.sum

    let sumSqSeq () = data |> Seq.map (fun x -> x * x) |> Seq.sum
    let sumSqArray () = data |> Array.map (fun x -> x * x) |> Array.sum
    let sumSqStreams () = Stream.ofArray data |> Stream.map (fun x -> x * x) |> Stream.sum    

    

    let log s f = 
        GC.Collect()
        GC.WaitForPendingFinalizers()
        GC.Collect()
        
        let sw = System.Diagnostics.Stopwatch.StartNew()
        let res = f()
        printfn "%s - Result %A - Completed in %s ms" s res (sw.ElapsedMilliseconds.ToString())

    log "sumSeq" sumSeq        
    log "sumArray" sumArray
    log "sumStreams" sumStreams

    log "sumSqSeq" sumSqSeq
    log "sumSqArray" sumSqArray
    log "sumSqStreams" sumSqStreams

    Console.ReadLine() |> ignore

    0 // return an integer exit code
