namespace Easj360FSharp

module LowLock =

    open System.Threading

    type LazyAux<'T> =
        | Value of 'T
        | Exception of exn
        | Thunk of (unit -> 'T)

    type Lazy<'T>(value: LazyAux<'T>) =
        let mutable value = value
        let sync = box 0    
        member private this.Value
          with get() =
            let x = value
            System.Threading.Thread.MemoryBarrier()
            x
          and set v =
            //System.Threading.Thread.MemoryBarrier()
            value <- v    
       
        member x.Force() =
          match x.Value with
          | Value x -> x
          | Exception e -> raise e
          | Thunk f ->
              lock sync (fun () ->
                match x.Value with
                | Value x -> x
                | Exception e -> raise e
                | Thunk f ->
                    try
                      let v = f()
                      x.Value <- Value v
                      v
                    with e ->
                      x.Value <- Exception e
                      raise e)

(*
> let mutable locks = 0;;
val mutable locks : int = 0> let lock x f =
    locks <- locks + 1
    lock x f;;
val lock : 'a -> (unit -> 'b) -> 'b when 'a : not struct

> type t =
    | Leaf
    | Branch of Lazy<t> * Lazy<t>;;
type t =
  | Leaf
  | Branch of Lazy<t> * Lazy<t>> let rec mk = function
    | 0 -> Lazy(Thunk(fun () -> Leaf))
    | n ->
        Lazy(Thunk(fun () ->
          let t = mk(n-1)
          Branch(t, t)));;
val mk : int -> Lazy<t>

let rec count (t: Lazy<_>) =
    match t.Force() with
    | Leaf -> 0
    | Branch(l, r) -> count l + 1 + count r;;
val count : Lazy<t> -> int

> let depth = 25;;
val depth : int = 25> do
    let t = System.Diagnostics.Stopwatch.StartNew()
    printf "Locking: %A\n" (mk depth |> count)
    printf "%d locks taken\n" locks
    locks <- 0
    printf "Took %gs\n" t.Elapsed.TotalSeconds;;
Locking: 33554431
67108863 locks taken
Took 12.7724s
val it : unit = ()

================================================

do
    let t = System.Diagnostics.Stopwatch.StartNew()
    printf "Boxed: %A\n" (mk depth |> count)
    printf "%d locks taken\n" locks
    locks <- 0
    printf "Took %gs\n" t.Elapsed.TotalSeconds;;
Boxed: 33554431
26 locks taken
Took 5.0688s
val it : unit = ()

================================================

do
    let t = System.Diagnostics.Stopwatch.StartNew()
    let tree = mk depth
    seq { for i in 1..8 ->
            async { return count tree } }
    |> Async.Parallel
    |> Async.RunSynchronously
    |> printf "Boxed: %A\n"
    locks <- 0
    printf "Took %gs\n" t.Elapsed.TotalSeconds;;
Boxed: [|33554431; 33554431; 33554431; 33554431; 33554431; 33554431; 33554431; 33554431|]
460824159 locks taken
Took 148.983s
val it : unit = ()

*)