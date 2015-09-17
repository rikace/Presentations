#load "..\Utilities\PSeq.fs"

open System.IO
open System
open System.Threading
open System.Threading.Tasks
open Microsoft.FSharp.Collections

let isPrime(n) =
    let top = int(sqrt(float(n)))
    let rec isPrimeUtil(i) =
        if i > top then true
        elif n % i = 0 then false
        else isPrimeUtil(i + 1)
    (n > 1) && isPrimeUtil(2)

// #time
[1000..10000000] 
    |> PSeq.filter isPrime 
    |> PSeq.length


[1000..10000000] |> List.filter isPrime |> List.length


let pfor nfrom nto f =
   Parallel.For(nfrom, nto + 1, Action<_>(f)) |> ignore
   
   
pfor 1000 10000000 (isPrime >> ignore)

[| 1000..10000000 |]
 |> Array.Parallel.map (isPrime)
 |> Array.toSeq 
 |> PSeq.filter(fun x -> x)
 |> PSeq.length

let arrPrimw, arrNotPrime = 
    [| 1000..10000000 |]
    |> Array.Parallel.partition(fun x -> isPrime(x))
arrPrimw |> Array.length


////////////// PARALLEL
let shortCircuitExample shortCircuit =
  let bag = System.Collections.Concurrent.ConcurrentBag<_>()
  Parallel.For(
    0, 999999, (fun i s -> if i < 10000 then bag.Add i else shortCircuit s)) |> ignore
  (bag, bag.Count)


Parallel.Invoke(
  (fun () -> printfn "Task 1"),
  (fun () -> Task.Delay(100).Wait()
             printfn "Task 2"),
  (fun () -> printfn "Task 3"))


//////////////  TEST PARALLEL VS SYNC ////////
let dims (a: float [,]) =
    Array2D.length1 a, Array2D.length2 a
  
let mk m n =
    Array2D.create m n 0.0
  
let test mul m n p =
    mul (mk m p) (mk p n) (mk m n)
  
module Serial =
    let mul a b (c: _ [,]) =
      let (am, an), (bm, bn) = dims a, dims b
      for i = 0 to am - 1 do
        for j = 0 to bn - 1 do
          for k = 0 to an - 1 do
            c.[i,j] <- c.[i,j] + a.[i,k] * b.[k,j]
    
test Serial.mul 1000 1000 1000
  
module Parallel =
    let parmul a b (c: float [,]) =
      let (an, am), (bn, bm) = dims a, dims b
      Tasks.Parallel.For(0, an, fun i ->
        Tasks.Parallel.For(0, bm, fun j ->
          for k = 0 to am - 1 do
            c.[i,j] <- c.[i,j] + a.[i,k] * b.[k,j])
        |> ignore)
      |> ignore
    
test Parallel.parmul 1000 1000 1000