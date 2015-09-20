#load "Stm.fs"
#load "..\CommonModule.fsx"

open System
open System.Threading
open DemoConcurrecny.Stm
open Common

(*  Transacted memory locations have the type TVar<'a>. 
    This is a reference cell like Ref<'a> in F# but its value can 
    only be read and written from inside a transaction. 
    A TVar is created using the newTVar function. *)
let x = newTVar 0  // val x : TVar<int>
let x' = ref 0  // but being a TVar instead of a Ref, 
                // the location x can be safely shared between multiple threads. 


(*  A transaction is defined using a computation expression. 
    A transaction is a value of type Stm<'a> which represents a computation 
    that when executed will make an atomic change to the values 
    of zero or more TVars and then return a value of type 'a. 
  
  The type system ensures at compile-time that all reads and writes 
  to a TVar only take place inside a transaction. 
  The functions readTVar and writeTVar read and write a TVar respectively. *)

// A function that increments the value stored in a TVar.
let incr x = 
  stm {
    let! v = readTVar x // the value of the location is read and stored in v
    do! writeTVar x (v+1) // The incremented value is then written back to the location
    return v } // v is returned as the result of the computation

//  val readTVar : TVar<'a> -> Stm<'a>
//  val writeTVar : TVar<'a> -> 'a -> Stm<unit>
//  val incr : TVar<int> -> Stm<int>


// When executed the read and write happen as an atomic block 
// which means logically that an interleaving thread won’t access the location between the read and write.

incr x |> atomically // val : atomically : Stm<'a> -> 'a
                     // val it : int = 0
incr x |> atomically // val it : int = 1

let ``quick Test Increment Atomically`` = 
    let worker () = 
        let result = (incr x) |> atomically        
        printfn "Thread %d Read & write x inc: %d" Thread.CurrentThread.ManagedThreadId result
        // Thread.Sleep(1)
    let spawn (f : unit -> unit) = let t = new Thread(f) in t.Start(); t
    for _ in [1..75] do spawn worker |> ignore


// An MVar is a mutable location like a TVar, except that it may be either empty, or full with a value
type MVar<'a> = TVar<option<'a> >
 
let newEmptyMVar () = newTVar None
let newFullMVar v = newTVar (Some v)
 
// The takeMVar function leaves a full MVar empty, and blocks on an empty MVar. 
// val takeMVar : MVar<'a> -> Stm<'a>
let takeMVar mv = 
  stm { let! v = readTVar mv
        return! match v with
                | Some a -> 
                    stm { do! writeTVar mv None
                          return a }
                | None -> retry () }
 
 // A putMVar on an empty MVar leaves it full, and blocks on a full MVar. 
 // val putMVar : MVar<'a> -> 'a -> Stm<unit>
let putMVar mv a =
  stm { let! v = readTVar mv
        return! match v with
                | None -> writeTVar mv (Some a)
                | Some _ -> retry () }

// The function tryTeakeMVar is the non-blocking version of takeMVar; 
// it returns an option value depending on whether a value was available.
let tryTakeMVar mv = 
  orElse 
    (stm { let! v = takeMVar mv
           return Some v })
    (stm { return None })


(********************************************************************************************************)
let test l1 l2 num_threads =
    let q1 = ListQueue.ofList l1
    let q2 = ListQueue.ofList l2
    let move_item q1 q2 =
        stm { let! x = ListQueue.dequeue q1
              do! ListQueue.enqueue q2 x 
              return x }
    let stop = newTVar false
    let rnd = new Random()
    let rec worker q1 q2 (fmt : string) =
        let x = 
            stm { let! stop' = readTVar stop
                  return! if not stop' 
                            then liftM Some (move_item q1 q2)
                            else stm.Return(None) } |> atomically
        match x with
        | Some x -> Console.WriteLine(fmt, Thread.CurrentThread.ManagedThreadId, x)
                    Thread.Sleep(rnd.Next(1000))
                    worker q1 q2 fmt
        | None -> ()
  
    let left_worker () = worker q1 q2 "Thread {0} moved item {1} left."
    let right_worker () = worker q2 q1 "Thread {0} moved item {1} right."
    let spawn (f : unit -> unit) = let t = new Thread(f) in t.Start(); t
    let threads = [ for _ in [1..num_threads] -> [spawn left_worker; spawn right_worker] ]
    let terminate () = 
        writeTVar stop true |> atomically
        threads |> Seq.concat |> Seq.iter (fun t -> t.Join()) 
        Console.WriteLine("Terminated.")
        stm { let! l1 = ListQueue.toList q1
              let! l2 = ListQueue.toList q2
              return l1,l2 } |> atomically
    terminate
    
let runTest () = 
    Console.WriteLine("Started.")
    let t = test [1..50] [51..100] 10 
    Thread.Sleep(3000)
    t () |> ignore

runTest()     


(********************************************************************************************************)

let testIncrVar n =     
    let x = newTVar 0
    let y = newTVar 0

    // STM does not define the Haskell's check function, so we define our own here
    // The retry operation is used to re-execute a transaction from the beginning
    let check (b: bool) = stm { if b then return () else return! retry() } 

    TimeMeasurement.stopTime <| fun() ->
        seq { 1..n } 
        |> mapM_ (fun i ->
            stm {
                let! x' = readTVar x
                let! y' = readTVar y
                do! check (x' = y')
                do! writeTVar x (x' + i)
                do! writeTVar y (y' + i) }) 
        |> atomically
    |> snd 
    |> printfn "Elapsed: %A ms"
    
testIncrVar 100


(********************************************************************************************************)

let enqueue (queue:ArrayQueue.Queue<_>) item =
  stm { let! used = readTVar queue.used
        return! if used < queue.len
                then stm { let! head = readTVar queue.head
                           do! writeTVar queue.a.[(head+used) % queue.len] item
                           return! writeTVar queue.used (used+1) }
                else retry () }

let (queue:ArrayQueue.Queue<int>) = ArrayQueue.ofList 6 [0..5]
(enqueue queue 6) |> atomically
 