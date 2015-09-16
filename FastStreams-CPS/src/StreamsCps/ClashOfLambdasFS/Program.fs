open System
open System.Linq
open System.Collections
open System.Collections.Generic
open System.Diagnostics
open Nessos.LinqOptimizer.FSharp
open Nessos.LinqOptimizer.Base
open Nessos.Streams
open LambdaMicrobenchmarking

type Ref(num : int) =
  member self.Num = num


[<EntryPoint>]
let main argv =
  //////////////////////////
  // Input initialization //
  //////////////////////////
  let N = 10000000
  let v = Enumerable.Range(1, N).Select(fun x -> (int64) (x % 1000)).ToArray()
  let vHi = Enumerable.Range(1, 1000000).Select(fun x -> (int64) x).ToArray()
  let vLow = Enumerable.Range(1, 10).Select(fun x -> (int64) x).ToArray()
  let refs = [|1..N|] |> Array.map (fun num -> new Ref(num))
  
  ///////////////////////////
  // Benchmarks definition //
  ///////////////////////////
  let sumBaseline () =
    let mutable acc = 0L
    for i=0 to v.Length-1 do
      acc <- v.[i] + acc
    acc

  let sumofSquaresBaseline () =
    let mutable acc = 0L
    for i=0 to v.Length-1 do
      acc <- v.[i] * v.[i] + acc
    acc

  let sumofSquaresEvenBaseline () =
    let mutable acc = 0L
    for i=0 to v.Length-1 do
      if v.[i] % 2L = 0L
      then
        acc <- v.[i] * v.[i] + acc
    acc

  let cartBaseline () =
    let mutable acc = 0L
    for d=0 to vHi.Length-1 do
      for dp=0 to vLow.Length-1 do
        acc <- vHi.[d] * vHi.[dp] + acc
    acc

  let refBaseline () =
    let mutable count = 0
    for i=0 to refs.Length-1 do
      if refs.[i].Num % 5 = 0 && refs.[i].Num % 7 = 0
      then
        count <- count + 1
    count

  let sumLinq () = Seq.sum v
  let sumLinqOpt = v |> Query.ofSeq |> Query.sum |> Query.compile
  let sumSqLinq () = v |> Seq.map (fun x -> x * x) |> Seq.sum
  let sumSqLinqOpt = v |> Query.ofSeq |> Query.map(fun x -> x * x) |> Query.sum |> Query.compile
  let sumSqEvenLinq () = v |> Seq.filter (fun x -> x % 2L = 0L) |> Seq.map (fun x -> x * x) |> Seq.sum
  let sumSqEvenLinqOpt = v |> Query.ofSeq |> Query.filter (fun x -> x % 2L = 0L) |> Query.map(fun x -> x * x) |> Query.sum |> Query.compile
  let cartLinq () = vHi |> Seq.collect (fun x -> Seq.map (fun y -> x * y) vLow) |> Seq.sum
  let cartLinqOpt = vHi |> Query.ofSeq |> Query.collect (fun x -> Seq.map (fun y -> x * y) vLow) |> Query.sum |> Query.compile
  let parallelSumLinq() = v.AsParallel().Sum()
  let parallelSumLinqOpt= v |> PQuery.ofSeq |> PQuery.sum |> PQuery.compile
  let parallelSumSqLinq() = v.AsParallel().Select(fun x -> x * x).Sum()
  let parallelSumSqLinqOpt = v |> PQuery.ofSeq |> PQuery.map(fun x -> x * x) |> PQuery.sum |> PQuery.compile
  let parallelSumSqEvenLinq () = v.AsParallel().Where(fun x -> x % 2L = 0L).Select(fun x -> x * x).Sum()
  let parallelSumSqEvenLinqOpt = v |> PQuery.ofSeq |> PQuery.filter (fun x -> x % 2L = 0L) |> PQuery.map(fun x -> x * x) |> PQuery.sum |> PQuery.compile
  let parallelCartLinq () = vHi.AsParallel().SelectMany(fun x -> vLow.Select(fun y -> x * y)).Sum()
  let parallelCartLinqOpt = vHi |> PQuery.ofSeq |> PQuery.collect (fun x -> Seq.map (fun y -> x * y) vLow) |> PQuery.sum |> PQuery.compile

  let refLinq () = refs |> Seq.filter (fun ref -> ref.Num % 5 = 0) |> Seq.filter (fun ref -> ref.Num % 7 = 0) |> Seq.length
  let refLinqOpt  = refs |> Query.ofSeq |> Query.filter (fun ref -> ref.Num % 5 = 0) |> Query.filter (fun ref -> ref.Num % 7 = 0) |> Query.length  |> Query.compile
  let parallelRefLinq () = refs.AsParallel().Where(fun (ref : Ref) -> ref.Num % 5 = 0).Where(fun (ref : Ref) -> ref.Num % 7 = 0).Count()
  let parallelRefLinqOpt = refs |> PQuery.ofSeq |> PQuery.filter(fun (ref : Ref) -> ref.Num % 5 = 0) |> PQuery.filter(fun (ref : Ref) -> ref.Num % 7 = 0) |> PQuery.length |> PQuery.compile

  let sumStreams() = v |> Stream.ofArray |> Stream.sum
  let sumSqStreams() = v |> Stream.ofArray |> Stream.map(fun x -> x * x) |> Stream.sum
  let sumSqEvenStreams() = v |> Stream.ofArray |> Stream.filter (fun x -> x % 2L = 0L) |> Stream.map(fun x -> x * x) |> Stream.sum
  let cartStreams() = vHi |> Stream.ofArray |> Stream.collect (fun x -> vLow |> Stream.ofArray |> Stream.map (fun y -> x * y) ) |> Stream.sum
  let refStream () = refs |> Stream.ofArray |> Stream.filter (fun ref -> ref.Num % 5 = 0) |> Stream.filter (fun ref -> ref.Num % 7 = 0) |> Stream.length

  //////////////////////////
  // Benchmarks execution //
  //////////////////////////

  let script = [|
    ("sumBaseline",  Func<int64>  sumBaseline);
    ("sumSeq",  Func<int64>  sumLinq);
    ("sumSeqOpt",  Func<int64>  sumLinqOpt);
    ("sumPar",  Func<int64>  parallelSumLinq );
    ("sumParOpt",  Func<int64>  parallelSumLinqOpt );
    ("sumOfSquaresBaseline",  Func<int64>  sumofSquaresBaseline);
    ("sumOfSquaresSeq",  Func<int64>  sumSqLinq );
    ("sumOfSquaresSeqOpt",  Func<int64>  sumSqLinqOpt );
    ("sumOfSquaresPar",  Func<int64>  parallelSumSqLinq );
    ("sumOfSquaresParOpt",  Func<int64>  parallelSumSqLinqOpt );
    ("sumOfSquaresEvenBaseline",  Func<int64>  sumofSquaresEvenBaseline);
    ("sumOfSquaresEvenSeq",  Func<int64>  sumSqEvenLinq );
    ("sumOfSquaresEvenSeqOpt",  Func<int64>  sumSqEvenLinqOpt );
    ("sumOfSquaresEvenPar",  Func<int64>  parallelSumSqEvenLinq );
    ("sumOfSquaresEvenParOpt",  Func<int64>  parallelSumSqEvenLinqOpt );
    ("cartBaseline",  Func<int64>  cartBaseline);
    ("cartSeq",  Func<int64>  cartLinq);
    ("cartSeqOpt",  Func<int64>  cartLinqOpt);
    ("cartPar",  Func<int64>  parallelCartLinq);
    ("cartParOpt",  Func<int64>  parallelCartLinqOpt)|] |> fun x -> Script.Of x

  let refScript = [|
    ("refBaseline",  Func<int> refBaseline);
    ("refSeq",  Func<int> refLinq);
    ("refSeqOpt", Func<int> refLinqOpt);
    ("refPar",  Func<int> parallelRefLinq);
    ("refParOpt",  Func<int> parallelRefLinqOpt)|] |> fun x -> Script.Of x

  let streamsScript = [|
    ("sumStreams",  Func<int64> sumStreams);
    ("sumSqStreams",  Func<int64> sumSqStreams);
    ("sumSqEvenStreams", Func<int64> sumSqEvenStreams);
    ("cartStreams",  Func<int64> cartStreams);|] |> fun x -> Script.Of x

  script.WithHead() |> ignore
  script.RunAll() |> ignore
  refScript.RunAll() |> ignore
  streamsScript.RunAll() |> ignore
  
  0 
