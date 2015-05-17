open System.Threading

let sumArray (arr : int[]) =
    let total = ref 0
    let thread1Finished = ref false
    ThreadPool.QueueUserWorkItem(
        fun _ -> for i = 0 to arr.Length / 2 - 1 do
                    total := arr.[i] + !total
                 thread1Finished := true
        ) |> ignore

    let thread2Finished = ref false

    ThreadPool.QueueUserWorkItem(
        fun _ -> for i = 0 to arr.Length / 2 - 1 do
                    total := arr.[i] + !total
                 thread2Finished := true
        ) |> ignore
    
    while !thread1Finished = false ||
          !thread2Finished = false do
          Thread.Sleep(0)
    !total

let lockedSumArray (arr : int[]) =
    let total = ref 0
    let thread1Finished = ref false
    ThreadPool.QueueUserWorkItem(
        fun _ -> for i = 0 to arr.Length / 2 - 1 do
                    lock (total) (fun () -> total := arr.[i] + !total)
                 thread1Finished := true
        ) |> ignore

    let thread2Finished = ref false

    ThreadPool.QueueUserWorkItem(
        fun _ -> for i = arr.Length / 2 to arr.Length - 1 do
                    lock (total) (fun () -> total := arr.[i] + !total)
                 thread2Finished := true
        ) |> ignore

    while !thread1Finished = false ||
          !thread2Finished = false do
          Thread.Sleep(0)
    !total

let millionOnes = Array.create 1000000 1
// Sum must be 1000000

sumArray millionOnes 

lockedSumArray millionOnes  


////////// ISOLATE //////////

let recSumArray (arr : int[]) =
    let sum (ints:int []) = async {
        let total = ref 0
        for i = 0 to ints.Length - 1 do
            total := !total + ints.[i]
        return !total }

    let sum' (ints:int[]) =
        let rec sumrec (ints:int []) index total = async {
            match index with 
            | n when n = ints.Length -> return total
            | i -> return! sumrec ints (index + 1) (total + ints.[index]) }
        sumrec ints 0 0

    let partition = arr.Length / 2
    let sumFirstPartition = sum' arr.[0..partition - 1] 
    let sumSecondPartition = sum' arr.[partition..arr.Length - 1]

    let sumResult = Async.Parallel [sumFirstPartition; sumSecondPartition] 
                    |> Async.RunSynchronously
    sumResult.[0] + sumResult.[1]
    //sumResult

recSumArray millionOnes
