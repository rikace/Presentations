namespace  Utilities

open System
open ParallelSeq

[<RequireQualifiedAccess>]
module SimpleAlghos =

    let isPrime n =
      let max = int (Math.Sqrt( float n ))
      let anydiv = { 2 .. max } |> Seq.filter ( fun d -> n%d = 0) |> (not << Seq.isEmpty)
      not ((n = 1) || anydiv)


    
    let benchmark name f = 
        printfn "%s" name
        let sw = System.Diagnostics.Stopwatch.StartNew()
        f() |> ignore
        sw.ElapsedMilliseconds
        
    let list_interval = [1 .. 1000000] 
    let c = benchmark ("Standard calculation") ( fun () ->
        list_interval 
        |> List.filter isPrime |> List.length)
    Console.WriteLine("Time elapsed for calculate prime number 1 to 1M: {0}", c) 

    let p = benchmark ("Patallel calculation") ( fun () ->
        list_interval 
        |> PSeq.filter isPrime |> PSeq.length)
    Console.WriteLine("Time elapsed for calculate prime number 1 to 1M: {0}", p) 
      