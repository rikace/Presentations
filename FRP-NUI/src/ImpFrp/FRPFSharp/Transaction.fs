namespace FRPFSharp

open System

type Runnable = unit -> unit

type private Entry(rank : Node, action : Handler<Atomically>) = 
    let compare (a : IComparable<'a>) (b : 'a) = a.CompareTo(b)
    static let nextSeq = ref 0L
    
    let seq = 
        let s = !nextSeq
        nextSeq := (!nextSeq) + 1L
        s
    
    member this.Rank = rank
    member this.Seq = seq
    member this.Action = action
    interface IComparable with
        member this.CompareTo(o : obj) = 
            let e = o :?> Entry
            match compare this.Rank e.Rank with
            | 0 -> 
                if seq < e.Seq then 1
                elif seq > e.Seq then -1
                else 0
            | n -> n

and Atomically() = 
    let prioritizedQ = PriorityQueue<Entry>()
    let entries = System.Collections.Generic.HashSet<Entry>()
    let lastQ = new System.Collections.Generic.List<Runnable>()
    let postQ = new System.Collections.Generic.List<Runnable>()
    static let currentTransaction : Atomically option ref = ref None
    static member GetCurrentTransaction() = lock Atomically.TransactionLock (fun () -> !currentTransaction)
    
    static member Run<'a>(code : Func<'a>) = 
        lock Atomically.TransactionLock (fun () -> 
            let oldTransaction = !currentTransaction
            try 
                if (!currentTransaction).IsNone then currentTransaction := Some(Atomically())
                code.Invoke()
            finally
                if oldTransaction.IsNone then (!currentTransaction).Value.Close()
                currentTransaction := oldTransaction)
    
    static member RunVoid(code : Runnable) = Atomically.Run(code)
    
    static member Apply<'a>(code : Func<Atomically, 'a>) = 
        lock Atomically.TransactionLock (fun () -> 
            let oldTransaction = !currentTransaction
            try 
                if (!currentTransaction).IsNone then currentTransaction := Some(Atomically())
                code.Invoke((!currentTransaction).Value)
            finally
                if oldTransaction.IsNone then (!currentTransaction).Value.Close()
                currentTransaction := oldTransaction)
    
    static member Run(code : Handler<Atomically>) = Atomically.Apply(fun t -> code.Run(t))
    static member TransactionLock = obj()
    static member ListenersLock = obj()
    member val ToRegen = false with get, set
    
    member this.Prioritized(rank : Node, action : Handler<Atomically>) = 
        let e = Entry(rank, action)
        prioritizedQ.Enqueue(e)
        entries.Add(e) |> ignore
    
    member this.Prioritized(rank : Node, action : Atomically -> unit) = 
        this.Prioritized(rank, 
                         { Fired = false
                           Run = action })
    
    member this.Last(action : Runnable) = lastQ.Add(action)
    //    member this.Last(action: Action) =
    //        lastQ.Add(fun () -> action.Invoke())
    member this.Post(action : Runnable) = postQ.Add(action)
    
    //    member this.Post(action: Action) =
    //        postQ.Add(fun () -> action.Invoke())
    member private this.CheckRegen() = 
        if this.ToRegen then 
            this.ToRegen <- false
            prioritizedQ.Clear()
            for e in entries do
                prioritizedQ.Enqueue(e)
    
    member this.Close() = 
        this.CheckRegen()
        while prioritizedQ.Count <> 0 do
            let e = prioritizedQ.Dequeue()
            entries.Remove(e) |> ignore
            e.Action.Run(this)
        for action in lastQ do
            action()
        lastQ.Clear()
        for action in postQ do
            action()
        postQ.Clear()
