namespace FRPFSharp

open System

type PriorityQueue<'a when 'a :> IComparable>() = 
    let capacity = ref 15
    let count = ref 0
    let version = ref 0
    let heap = ref (Array.zeroCreate<'a> !capacity)

    let getLeftChild index = (index * 2) + 1
    let getParent index = (index - 1) / 2

    let bubbleUp index item =
        let parent = ref (getParent index)
        let index = ref index

        //note: (index > 0) means there is a parent
        while (!index > 0) &&
              ((!heap).[!parent].CompareTo(item) < 0) do
            (!heap).[!index] <- (!heap).[!parent]
            index := !parent
            parent := getParent !index

        (!heap).[!index] <- item

    let trickleDown (index : int) item =
        let index = ref index
        let child = ref (getLeftChild !index)

        while !child < !count do
            if ((!child + 1) < !count) && 
               ((!heap).[!child].CompareTo((!heap).[!child + 1]) < 0) then
                child := !child + 1

            (!heap).[!index] <- (!heap).[!child]
            index := !child
            child := getLeftChild !index
        
        bubbleUp !index item

    let growHeap () =
        capacity := (!capacity * 2) + 1
        Array.Resize(heap, !capacity)

    member this.Dequeue () = 
        if !count = 0 then
            raise (new InvalidOperationException())

        let result = (!heap).[0]
        decr count
        trickleDown 0 (!heap).[!count]
        (!heap).[!count] <- Unchecked.defaultof<'a>
        incr version
        result

    member this.Enqueue(item: 'a) =
        if obj.ReferenceEquals(null, item) then
            raise (new ArgumentNullException("item"))

        if !count = !capacity then
            growHeap()

        incr count
        bubbleUp (!count - 1) item
        incr version

    member this.Count = !count

    member this.Clear() =
        capacity := 15
        heap := Array.zeroCreate (!capacity)
        count := 0
        version := 0
