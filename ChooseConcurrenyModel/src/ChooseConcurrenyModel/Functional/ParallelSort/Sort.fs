namespace ParallelSort

open System
open System.Threading.Tasks

module Sort =
    /// Array length to use InsertionSort instead of SequentialQuickSort
    let mutable threshold = 150

    /// Swap elements of the array at the specified indices
    let swap (array:_[]) i j = 
        let temp = array.[i]
        array.[i] <- array.[j]
        array.[j] <- temp

    /// Insertion sort which is more efficient for small arrays
    /// (it is used when the length is smaller than 'threshold')
    let insertionSort (array:int[]) fromIndex toIndex =
        for i in fromIndex + 1 .. toIndex - 1 do 
            let a = array.[i]
            let mutable j = i - 1
            while j >= fromIndex && array.[j] > a do
                array.[j + 1] <- array.[j]
                j <- j - 1
            array.[j + 1] <- a


    let partition (array:int[]) fromIndex toIndex pivotIndex =
        // Pre: from <= pivot < to (other than that, pivot is arbitrary)
        let arrayPivot = array.[pivotIndex]  
        // move pivot value to end for now, after this pivot not used
        swap array pivotIndex (toIndex - 1)
        let mutable newPivot = fromIndex
        // be careful to leave pivot value at the end
        for i in fromIndex .. toIndex - 2 do
            // Invariant: from <= newpivot <= i < to - 1 && 
            //   forall from <= j <= newpivot, array[j] <= arrayPivot &&
            //   forall newpivot < j <= i, array[j] > arrayPivot
            if array.[i] <= arrayPivot then 
                // move value smaller than arrayPivot down to newPivot
                swap array newPivot i
                newPivot <- newPivot + 1
        // move pivot value to its final place
        swap array newPivot (toIndex - 1)
        newPivot 
        // Post: forall i <= newpivot, array[i] <= array[newpivot]  && forall i > ...

    // --------------------------------------------------------------------------
    // Sequential & parallel implementation 
    // --------------------------------------------------------------------------
    
    /// Sequential implementation of in-place QuickSort algorithm
    let sequentialQuickSort array = 
        // Recursive sequential QuickSort
        let rec sortUtil array fromIndex toIndex =
            if toIndex - fromIndex <= threshold then 
                insertionSort array fromIndex toIndex
            else
                // could be anything, use middle
                let pivot = fromIndex + (toIndex - fromIndex) / 2 
                let pivot = partition array fromIndex toIndex pivot
                // Assert: forall i < pivot, array[i] <= array[pivot]  && forall i > ...
                sortUtil array fromIndex pivot
                sortUtil array (pivot + 1) toIndex
        
        // Start the recursion for the entire array
        sortUtil array 0 array.Length


    /// Parallel implementation of in-place QuickSort algorithm (using Tasks)
    let parallelQuickSort array =
        // Recursive parallel QuickSort
        let rec sortUtil array fromIndex toIndex depthRemaining =
            if toIndex - fromIndex <= threshold then
                insertionSort array fromIndex toIndex
            else
                // could be anything, use middle
                let pivot = fromIndex + (toIndex - fromIndex) / 2 
                let pivot = partition array fromIndex toIndex pivot
                if depthRemaining > 0 then
                    Parallel.Invoke
                      ( new Action(fun () -> 
                          sortUtil array fromIndex pivot (depthRemaining - 1)),
                        new Action(fun () -> 
                          sortUtil array (pivot + 1) toIndex (depthRemaining - 1)) )
                else
                    sortUtil array fromIndex pivot 0
                    sortUtil array (pivot + 1) toIndex 0

        // Start the recursion for the entire array
        sortUtil array 0 array.Length (int(Math.Log(float Environment.ProcessorCount, 2.0) + 4.0))
