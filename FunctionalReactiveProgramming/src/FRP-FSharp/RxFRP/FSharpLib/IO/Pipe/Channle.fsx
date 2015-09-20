// http://t0yv0.blogspot.com/2011/12/making-async-5x-faster.html

#if INTERACTIVE
#else
namespace IntelliFactory.Examples
#endif

open System
open System.Collections.Concurrent
open System.Collections.Generic
open System.Threading
open System.Threading.Tasks

[<Sealed>]
type Pool private () =
    let queue = new BlockingCollection<_>(ConcurrentBag())
    let work () = while true do queue.Take()()
    let long = TaskCreationOptions.LongRunning
    let task = Task.Factory.StartNew(work, long)
    static let self = Pool()
    member private this.Add f = queue.Add f
    static member Spawn(f: unit -> unit) = self.Add f

[<AutoOpen>]
module FastAsync =
    type Async<'T> = ('T -> unit) -> unit

    [<Sealed>]
    type Async() =
        member inline this.Return(x: 'T) : Async<'T> =
            fun f -> f x
        member inline this.ReturnFrom(x: Async<'T>) = x
        member inline this.Bind
            (x: Async<'T1>, f: 'T1 -> Async<'T2>) : Async<'T2> =
            fun k -> x (fun v -> f v k)
        static member inline Start(x: Async<unit>) =
            Pool.Spawn(fun () -> x ignore)
        static member inline RunSynchronously(x: Async<'T>) : 'T =
            let res = ref Unchecked.defaultof<_>
            use sem = new SemaphoreSlim(0)
            Pool.Spawn(fun () ->
                x (fun v ->
                    res := v
                    ignore (sem.Release())))
            sem.Wait()
            !res
        static member inline FromContinuations
            (f: ('T -> unit) *
                (exn -> unit) *
                (OperationCanceledException -> unit) -> unit)
            : Async<'T> =
            fun k -> f (k, ignore, ignore)

    let async = Async()

[<Sealed>]
type Channel<'T>() =
    let readers = Queue()
    let writers = Queue()

    member this.Read ok =
        let task =
            lock readers <| fun () ->
                if writers.Count = 0 then
                    readers.Enqueue ok
                    None
                else
                    Some (writers.Dequeue())
        match task with
        | None -> ()
        | Some (value, cont) ->
            Pool.Spawn cont
            ok value

    member this.Write(x: 'T, ok) =
        let task =
            lock readers <| fun () ->
                if readers.Count = 0 then
                    writers.Enqueue(x, ok)
                    None
                else
                    Some (readers.Dequeue())
        match task with
        | None -> ()
        | Some cont ->
            Pool.Spawn ok
            cont x

    member inline this.Read() =
        Async.FromContinuations(fun (ok, _, _) ->
            this.Read ok)

    member inline this.Write x =
        Async.FromContinuations(fun (ok, _, _) ->
            this.Write(x, ok))

module Main =
    let test (n: int) =
        let chan = Channel()
        let rec writer (i: int) =
            async {
                if i = 0 then
                    return! chan.Write 0
                else
                    do! chan.Write i
                    return! writer (i - 1)
            }
        let rec reader sum =
            async {
                let! x = chan.Read()
                if x = 0
                then return sum
                else return! reader (sum + x)
            }
        Async.Start(writer n)
        let clock = System.Diagnostics.Stopwatch()
        clock.Start()
        let r = Async.RunSynchronously(reader 0)
        stdout.WriteLine("Hops per second: {0}",
            float n / clock.Elapsed.TotalSeconds)
        r

    [<EntryPoint>]
    let main args =
        test 1000000
        |> printfn "Result: %i"
        0

#if INTERACTIVE
#time
Main.test 1000000
#endif

