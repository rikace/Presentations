#load "..\..\DemoConcurrency\DemoConcurrency\CommonModule.fsx"
open Common

// Memoization Sample

open System
open System.Collections.Generic

let memoizeG f =
    let cache = Dictionary<_, _>()
    fun x ->
        if cache.ContainsKey(x) then cache.[x]
        else let res = f x
             cache.[x] <- res
             res


let memoize f =
    let cache = ref Map.empty 
    fun x ->
        match (!cache).TryFind(x) with
        | Some res -> res
        | None ->
             let res = f x
             cache := (!cache).Add(x,res)
             res


let rec fibonacci =  
  fun n -> if n <= 2 then 1 else fibonacci(n - 1) + fibonacci(n - 2)

let fibonacciMemoized = memoize fibonacci
let result = fibonacciMemoized 35


let runFiboacci() =
        let result1 = fibonacci 35
        let result2 = fibonacci 35
        let result3 = fibonacci 35
        let result4 = fibonacci 35
        ()

let runFiboacciMem() =
        let result1 = fibonacciMemoized 35
        let result2 = fibonacciMemoized 35
        let result3 = fibonacciMemoized 35
        let result4 = fibonacciMemoized 35
        ()

benchmark (fun _ -> runFiboacci())

benchmark (fun _ -> runFiboacciMem())