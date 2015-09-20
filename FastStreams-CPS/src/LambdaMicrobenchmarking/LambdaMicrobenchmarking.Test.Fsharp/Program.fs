open System
open System.Linq
open System.Collections
open System.Collections.Generic
open LambdaMicrobenchmarking

[<EntryPoint>]
let main argv = 
    let N = 10000000
    let v = Enumerable.Range(1, N).Select(fun x -> (int64) (x % 1000)).ToArray()
    let vHi = Enumerable.Range(1, 10000).Select(fun x -> (int64) x).ToArray()
    let vLow = Enumerable.Range(1, 1000).Select(fun x -> (int64) x).ToArray()

    let sumLinq () = Seq.sum v
    let sumSqLinq () = v |> Seq.map (fun x -> x * x) |> Seq.sum
    let sumSqEvenLinq () = v |> Seq.filter (fun x -> x % 2L = 0L) |> Seq.map (fun x -> x * x) |> Seq.sum
    let cartLinq () = vHi |> Seq.collect (fun x -> Seq.map (fun y -> x * y) vLow) |> Seq.sum

    let script = [|
        ("sumLinq", Func<int64> sumLinq); 
        ("sumOfSquaresLinq", Func<int64> sumSqLinq);
        ("sumOfSquaresEvenLinq", Func<int64> sumSqEvenLinq);
        ("cartLinq", Func<int64> cartLinq)|] |> fun x -> Script.Of x
        
    let script2 = [|
        ("sumLinq", Func<int64> sumLinq); 
        ("sumOfSquaresLinq", Func<int64> sumSqLinq);
        ("sumOfSquaresEvenLinq", Func<int64> sumSqEvenLinq);
        ("cartLinq", Func<int64> cartLinq)|] |> fun x -> Script.Of x

    script.WithHead()|> ignore  
    script.RunAll() |> ignore

    script2.RunAll() |> ignore

    Script.Of("sumLinq", Func<int64> sumLinq)
          .Of("sumOfSquaresLinq", Func<int64> sumSqLinq)
          .Of("sumOfSquaresEvenLinq", Func<int64> sumSqEvenLinq)
          .Of("cartLinq", Func<int64> cartLinq)
          .WithHead()
          .RunAll |> ignore

    0
