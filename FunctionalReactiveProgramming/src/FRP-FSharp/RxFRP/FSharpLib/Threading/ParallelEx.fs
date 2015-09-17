namespace Easj360FSharp 

open System
open System.Collections.Concurrent
open System.Collections.Generic
open System.Linq
open System.Threading
open System.Threading.Tasks

module Threading =

    // parallel execution method
    type ParallelEx =

        // Parallel Invoke wrapper
        static member inline Invoke (actions:(unit -> unit) array) =
            match actions with
            | [||] -> ()
            | [|action|] -> action()
            | _ ->
                Array.map (fun x -> Action(x)) actions
                |> Parallel.Invoke   
        
        // TPL Warppers
        static member inline Do (actions:(unit -> unit) array, ?cancellationToken:CancellationToken) =
            match actions with
            | [||] -> ()
            | [|action|] -> action()
            | _ ->
                let tasks = 
                    match cancellationToken with
                    | Some c -> Array.map (fun x -> Task.Factory.StartNew(Action(x), c)) actions
                    | None -> Array.map (fun x -> Task.Factory.StartNew(Action(x))) actions
                Task.WaitAll tasks

        static member inline Let (funcs:(unit -> 'T) array, ?cancellationToken:CancellationToken) =
            match funcs with
            | [||] -> Array.empty
            | [|func|] -> [| func() |]
            | _ ->
                let tasks = 
                    match cancellationToken with
                    | Some c -> Array.map (fun x -> Task.Factory.StartNew(Func<'T>(x), c)) funcs
                    | None -> Array.map (fun x -> Task.Factory.StartNew(Func<'T>(x))) funcs
                Task.WaitAll [| for task in tasks -> task :> Task |]
                Array.map (fun (t: Task<'T>) -> t.Result) tasks

type private MergeArrayType =
    | FromArray
    | ToArray

type ParallelMergeSort() =    
    
    static member public Sort(array: 'T []) = 
        let arraySort = Array.copy array
        ParallelMergeSort.SortInPlaceInternal(arraySort)
        arraySort
    
    static member public SortBy(array: 'T [], projection: 'T -> 'Key) = 
        let arraySort = Array.copy array
        ParallelMergeSort.SortInPlaceInternal(array, projection = projection)
        arraySort

    static member public SortWith(array: 'T [], comparer: 'T -> 'T -> int) =
        let arraySort = Array.copy array
        ParallelMergeSort.SortInPlaceInternal(array, comparer = comparer)
        arraySort

    static member public SortInPlace(array: 'T []) = 
        ParallelMergeSort.SortInPlaceInternal(array)
    
    static member public SortInPlaceBy(array: 'T [], projection: 'T -> 'Key) =
        ParallelMergeSort.SortInPlaceInternal(array, projection = projection)

    static member public SortInPlaceWith(array: 'T [], comparer: 'T -> 'T -> int) = 
        ParallelMergeSort.SortInPlaceInternal(array, comparer = comparer)
   
    // Private function that is used to control the sorting
    static member private SortInPlaceInternal(array: 'T [], ?comparer: 'T -> 'T -> int, ?projection: 'T -> 'Key) =

        // used to do the merge and sort comparisions
        let sortComparer =
            match comparer with
            | Some c -> ComparisonIdentity.FromFunction c
            | _ -> ComparisonIdentity.Structural<'T>

        let projectionComparer = ComparisonIdentity.Structural<'Key>

        let inline sortComparerResult (item1: 'T) (item2: 'T) = 
            match projection with
            | Some p -> projectionComparer.Compare(p item1, p item2)
            | None -> sortComparer.Compare(item1, item2)

        // The merge of the two array
        let merge (toArray: 'T []) (fromArray: 'T []) (low1: int) (low2: int) (high1: int) (high2: int) =
            let mutable ptr1 = low1
            let mutable ptr2 = high1

            for ptr in low1..high2 do 
                if (ptr1 > low2) then
                    toArray.[ptr] <- fromArray.[ptr2]
                    ptr2 <- ptr2 + 1
                elif (ptr2 > high2) then
                    toArray.[ptr] <- fromArray.[ptr1]
                    ptr1 <- ptr1 + 1
                elif ((sortComparerResult fromArray.[ptr1] fromArray.[ptr2]) <= 0) then
                    toArray.[ptr] <- fromArray.[ptr1]
                    ptr1 <- ptr1 + 1
                else
                    toArray.[ptr] <- fromArray.[ptr2]
                    ptr2 <- ptr2 + 1

        // define the sort operation
        let parallelSort (array: 'T []) =              

            // control flow parameters
            let totalWorkers = int (2.0 ** float (int (Math.Log(float Environment.ProcessorCount, 2.0)))) 
            let auxArray : 'T array = Array.zeroCreate array.Length
            let workers : Task array = Array.zeroCreate (totalWorkers - 1)
            let iterations = int (Math.Log((float totalWorkers), 2.0))

            // define a key array if needed for sorting on a projection
            let keysArray = 
                match projection with
                | Some p -> Array.Parallel.init array.Length (fun idx -> p array.[idx])
                | None -> [||]

            // Number of elements for each array, if the elements number is not divisible by the workers
            // the remainders will be added to the first worker (the main thread)
            let partitionSize = ref (int (array.Length / totalWorkers))
            let remainder = array.Length % totalWorkers

            // Define the arrays references for processing as they are swapped during each iteration
            let swapped = ref false

            let inline getMergeArray (arrayType: MergeArrayType) =
                match (arrayType, !swapped) with
                | (FromArray, true) -> auxArray
                | (FromArray, false) -> array
                | (ToArray, true) -> array
                | (ToArray, false) -> auxArray

            use barrier = new Barrier(totalWorkers, fun (b) -> 
                partitionSize := !partitionSize <<< 1
                swapped := not !swapped)

            // action to perform the sort an merge steps
            let action (index: int) =   
                         
                //calculate the partition boundary
                let low = index * !partitionSize + match index with | 0 -> 0 | _ -> remainder
                let high = (index + 1) * !partitionSize - 1 + remainder

                // Sort the specified range - could implement QuickSort here
                let sortLen = high - low + 1
                match (comparer, projection) with
                | (Some _, _) -> Array.Sort(array, low, sortLen, sortComparer)
                | (_, Some p) -> Array.Sort(keysArray, array, low, sortLen)
                | (_, _) -> Array.Sort(array, low, sortLen)

                barrier.SignalAndWait()

                let rec loopArray loopIdx actionIdx loopHigh = 
                    if loopIdx < iterations then                                  
                        if (actionIdx % 2 = 1) then
                            barrier.RemoveParticipant()  
                        else
                            let newHigh = loopHigh + !partitionSize / 2
                            merge (getMergeArray FromArray) (getMergeArray ToArray) low loopHigh (loopHigh + 1) newHigh
                            barrier.SignalAndWait()
                            loopArray (loopIdx + 1) (actionIdx >>> 1) newHigh
                loopArray 0 index high

            for index in 1 .. workers.Length do
                workers.[index - 1] <- Task.Factory.StartNew(fun() -> action index)

            action 0

            // if odd iterations return auxArray otherwise array (swapped will be false)
            if not (iterations % 2 = 0) then  
                Array.blit auxArray 0 array 0 array.Length

        // Perform the sorting
        match array with
        | [||] -> failwith "Empty Array"
        | small when small.Length < (Environment.ProcessorCount * 2) ->
            match (comparer, projection) with
            | (Some c, _) -> Array.sortInPlaceWith c array
            | (_, Some p) -> Array.sortInPlaceBy p array
            | (_, _) -> Array.sortInPlace array
        | _ -> parallelSort array

type ParallelQuickSort() =
   
    static member public Sort(array: 'T []) = 
        let arraySort = Array.copy array       
        ParallelQuickSort.SortInPlaceInternal(arraySort)
        arraySort
    
    static member public SortBy(array: 'T [], projection: 'T -> 'Key) = 
        let arraySort = Array.copy array
        ParallelQuickSort.SortInPlaceInternal(array, projection = projection)
        arraySort

    static member public SortWith(array: 'T [], comparer: 'T -> 'T -> int) =
        let arraySort = Array.copy array
        ParallelQuickSort.SortInPlaceInternal(array, comparer = comparer)
        arraySort

    static member public SortInPlace(array: 'T []) = 
        ParallelQuickSort.SortInPlaceInternal(array)
    
    static member public SortInPlaceBy(array: 'T [], projection: 'T -> 'Key) =
        ParallelQuickSort.SortInPlaceInternal(array, projection = projection)

    static member public SortInPlaceWith(array: 'T [], comparer: 'T -> 'T -> int) = 
        ParallelQuickSort.SortInPlaceInternal(array, comparer = comparer)

    // counter for the degree of paallism
    static member private CurrentDop = ref 0
    static member private TargetDop = Environment.ProcessorCount * 2
   
    // Private function that is used to control the sorting
    static member private SortInPlaceInternal(array: 'T [], ?comparer: 'T -> 'T -> int, ?projection: 'T -> 'Key) =

        // definition of runtime parameters
        let smallThreshold = 32
        let parallelThreshold = 4 * 1024

        // define a key array if needed for sorting on a projection
        let keys = 
            match projection with
            | None -> [||]
            | Some p -> Array.Parallel.init array.Length (fun idx -> p array.[idx])

        // used to do the partition and sort comparisions
        let sortComparer =
            match comparer with
            | None -> ComparisonIdentity.Structural<'T>
            | Some c -> ComparisonIdentity.FromFunction c

        let projectionComparer =
            ComparisonIdentity.Structural<'Key>

        // swap elements (and maybe keys)
        let inline comparerResult left right =
            match projection with
            | None -> sortComparer.Compare(array.[left], array.[right])
            | Some _ -> projectionComparer.Compare(keys.[left], keys.[right])

        let inline swap x y =
            match projection with
            | None -> 
                let ae = array.[x]
                array.[x] <- array.[y]
                array.[y] <- ae
            | Some _ ->
                let ae = array.[x]
                array.[x] <- array.[y]
                array.[y] <- ae
                let ak = keys.[x]
                keys.[x] <- keys.[y]
                keys.[y] <- ak

        // sort three elements
        let inline sortThree low middle high = 
            if (comparerResult middle low < 0) then
                swap middle low
            if (comparerResult high middle < 0) then
                swap high middle
                if (comparerResult middle low < 0) then
                    swap middle low                                       

        // perform an in place partition with pivot in position low
        // taking average of 3 rather than -> swap low pivot 
        let inline partition (low:int) (high:int) =                            
            let pivot = (low + high) / 2    
            sortThree pivot low high            

            let mutable last = low
            for current in (low + 1)..high do 
                if (comparerResult current low < 0) then
                    last <- last + 1
                    swap last current

            swap low last
            last        

        // define the sort operation using Parallel.Invoke for a depth
        let rec quickSortDepth (low:int) (high:int) (depth:int) =
            let sortLen = high - low + 1
            match sortLen with
            | 0 | 1 -> ()
            | 2 -> if (comparerResult high low < 0) then swap high low
            | small when small < smallThreshold ->
                match (comparer, projection) with
                | (Some _, _) -> Array.Sort(array, low, sortLen, sortComparer)
                | (_, Some _) -> Array.Sort(keys, array, low, sortLen)
                | (_, _) -> Array.Sort(array, low, sortLen)
            | _ -> 
                let pivot = partition low high
                if (depth > 0 && sortLen > parallelThreshold) then
                    Threading.ParallelEx.Do [|
                        fun () -> quickSortDepth low (pivot - 1) (depth - 1);
                        fun () -> quickSortDepth (pivot + 1) high (depth - 1) |]
                else
                    quickSortDepth low (pivot - 1) 0
                    quickSortDepth (pivot + 1) high 0

        // define the sort operation using Parallel.Invoke for a count
        let rec quickSortCount (low:int) (high:int) =
            let sortLen = high - low + 1
            match sortLen with
            | 0 | 1 -> ()
            | 2 -> if (comparerResult high low < 0) then swap high low
            | small when small < smallThreshold ->
                match (comparer, projection) with
                | (Some _, _) -> Array.Sort(array, low, sortLen, sortComparer)
                | (_, Some _) -> Array.Sort(keys, array, low, sortLen)
                | (_, _) -> Array.Sort(array, low, sortLen)
            | _ -> 
                let pivot = partition low high
                if (!ParallelQuickSort.CurrentDop < ParallelQuickSort.TargetDop && sortLen > parallelThreshold) then
                    Interlocked.Increment(ParallelQuickSort.CurrentDop) |> ignore
                    Threading.ParallelEx.Do [|
                        fun () -> quickSortCount low (pivot - 1);
                        fun () -> quickSortCount (pivot + 1) high |]
                    Interlocked.Decrement(ParallelQuickSort.CurrentDop) |> ignore
                else
                    quickSortCount low (pivot - 1)
                    quickSortCount (pivot + 1) high

        // Perform the sorting
        // let targetDepth = int (Math.Log(float Environment.ProcessorCount, 2.0)) + 1
        // quickSortDepth 0 (array.Length - 1) targetDepth
        quickSortCount 0 (array.Length - 1)


module Array = 
    module Parallel = 

        let private smallThreshold = 48 * 1024
        let private largeThreshold = 2048 * 1024

        let sort (array: 'T []) = 
            match array.Length with
            | length when length > largeThreshold -> ParallelMergeSort.Sort(array)
            | length when length > smallThreshold -> ParallelQuickSort.Sort(array)
            | _ -> Array.sort array

        let sortBy (projection: 'T -> 'Key) (array: 'T []) = 
            match array.Length with
            | length when length > largeThreshold -> ParallelMergeSort.SortBy(array, projection)
            | length when length > smallThreshold -> ParallelQuickSort.SortBy(array, projection)
            | _ -> Array.sortBy projection array            

        let sortWith (comparer: 'T -> 'T -> int) (array: 'T []) = 
            match array.Length with
            | length when length > largeThreshold -> ParallelMergeSort.SortWith(array, comparer)
            | length when length > smallThreshold -> ParallelQuickSort.SortWith(array, comparer)
            | _ -> Array.sortWith comparer array             

        let sortInPlace (array: 'T []) = 
            match array.Length with
            | length when length > largeThreshold -> ParallelMergeSort.SortInPlace(array)
            | length when length > smallThreshold -> ParallelQuickSort.SortInPlace(array)
            | _ -> Array.sortInPlace array            

        let sortInPlaceBy (projection: 'T -> 'Key) (array: 'T []) = 
            match array.Length with
            | length when length > largeThreshold -> ParallelMergeSort.SortInPlaceBy(array, projection)
            | length when length > smallThreshold -> ParallelQuickSort.SortInPlaceBy(array, projection)
            | _ -> Array.sortInPlaceBy projection array            

        let sortInPlaceWith (comparer: 'T -> 'T -> int) (array: 'T []) = 
            match array.Length with
            | length when length > largeThreshold -> ParallelMergeSort.SortInPlaceWith(array, comparer)
            | length when length > smallThreshold -> ParallelQuickSort.SortInPlaceWith(array, comparer)
            | _ -> Array.sortInPlaceWith comparer array          

        module Merge =
            let sort (array: 'T []) = 
                ParallelMergeSort.Sort(array)

            let sortBy (projection: 'T -> 'Key) (array: 'T []) = 
                ParallelMergeSort.SortBy(array, projection)

            let sortWith (comparer: 'T -> 'T -> int) (array: 'T []) = 
                ParallelMergeSort.SortWith(array, comparer)

            let sortInPlace (array: 'T []) = 
                ParallelMergeSort.SortInPlace(array)

            let sortInPlaceBy (projection: 'T -> 'Key) (array: 'T []) = 
                ParallelMergeSort.SortInPlaceBy(array, projection)

            let sortInPlaceWith (comparer: 'T -> 'T -> int) (array: 'T []) = 
                ParallelMergeSort.SortInPlaceWith(array, comparer)

        module Quick =
            let sort (array: 'T []) = 
                ParallelQuickSort.Sort(array)

            let sortBy (projection: 'T -> 'Key) (array: 'T []) = 
                ParallelQuickSort.SortBy(array, projection)

            let sortWith (comparer: 'T -> 'T -> int) (array: 'T []) = 
                ParallelQuickSort.SortWith(array, comparer)

            let sortInPlace (array: 'T []) = 
                ParallelQuickSort.SortInPlace(array)

            let sortInPlaceBy (projection: 'T -> 'Key) (array: 'T []) = 
                ParallelQuickSort.SortInPlaceBy(array, projection)

            let sortInPlaceWith (comparer: 'T -> 'T -> int) (array: 'T []) = 
                ParallelQuickSort.SortInPlaceWith(array, comparer)

