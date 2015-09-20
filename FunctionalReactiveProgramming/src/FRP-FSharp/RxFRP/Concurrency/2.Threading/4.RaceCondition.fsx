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




