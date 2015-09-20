namespace Easj360FSharp

open System  
open System.Threading

module LockConcurrentQueue =

    type Lock<'a> = Lock of (obj -> 'a)

    let apply (Lock f) lock = f lock

    let mreturn x = Lock (fun lock -> x)

    let bind m f = Lock (fun lock -> apply (f (apply m lock)) lock)

    let run lock m =
      let lock = box lock
      Monitor.Enter(lock)
      try apply m lock
      finally Monitor.Exit(lock)

    let tryRun lock (timeout : int) m =
      let lock = box lock
      if Monitor.TryEnter(lock, timeout)
      then try Some (apply m lock)
           finally Monitor.Exit(lock)
      else None
    
    let wait = Lock (fun lock -> Monitor.Wait(lock) |> ignore)

    let tryWait (timeout : int) = Lock (fun lock -> Monitor.Wait(lock, timeout))

    let pulse = Lock (fun lock -> Monitor.Pulse(lock))

    let pulseAll = Lock (fun lock -> Monitor.PulseAll(lock))

    let getLock = Lock (fun lock -> lock)

    let tryWith m cth = Lock (fun lock -> try apply m lock with exc -> apply (cth exc) lock)

    let tryFinally m fin = Lock (fun lock -> try apply m lock finally apply fin lock)

    let liftM f m = bind m (f >> mreturn)

    type Builder () =
      member b.Return(x) = mreturn x
      member b.Bind(m, f) = bind m f
      member b.TryWith(m, cth) = tryWith m cth
      member b.TryFinally(m, fin) = Lock (fun lock -> try apply m lock finally fin ())
  
      member b.Let(x, f) = f x
      member b.Delay(f) = Lock (fun lock -> apply (f ()) lock)
      member b.Zero() = mreturn ()
      member b.Combine(m1, m2) = bind m1 (fun () -> m2)
  
      member b.BindUsing(m, f) = bind m (fun (x : #IDisposable) -> Lock (fun lock -> try apply (f x) lock finally x.Dispose()))
      member b.Using(x : #IDisposable, f) = Lock (fun lock -> try apply (f x) lock finally x.Dispose())

      member b.While(p, m) = Lock (fun lock -> while p () do apply m lock)
      member b.For(xs, f) = Lock (fun lock -> xs |> Seq.iter (fun x -> apply (f x) lock))
    
    let lock = Builder()

    let ifM cond th el = 
      Lock (fun lock -> if apply cond lock then apply th lock else apply el lock )

    let whileM p m = 
      Lock (fun lock -> while apply p lock do apply m lock )

    let map f xs =
      Lock (fun lock -> List.map (fun x -> apply (f x) lock) xs)

    let mapi f xs =
      Lock (fun lock -> List.mapi (fun i x -> apply (f i x) lock) xs)
  
    let iter f xs =
      Lock (fun lock -> List.iter (fun x -> apply (f x) lock) xs)

  
//
//open System.Collections.Generic
//
//let enqueue (q : Queue<_>) x = 
//  lock { if q.Count = 0 then return! pulseAll
//         return q.Enqueue(x) }
//   
//let dequeue (q : Queue<_>) = 
//  lock { while q.Count = 0 do return! wait
//         return q.Dequeue() }
//
//let myLock = new obj()
//
//let move q1 q2 = 
//  lock { let! x = dequeue q1 
//         do! enqueue q2 x
//         return x } |> run myLock


