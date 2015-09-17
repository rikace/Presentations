module Common


open System
open System.IO
open System.IO.Pipes
open System.Net
open System.ComponentModel
open System.Threading

//module StreamExt =
//   type Stream with
//      member x.ToObservable(size) =
//         Observable.Create(fun (observer: IObserver<_>) ->
//            let buffer = Array.zeroCreate size
//            let defered = Observable.Defer(fun () -> (x.ReadAsync (buffer, 0, size)).ToObservable())
//            Observable.Repeat<int>(defered)
//                      .Select(fun i -> buffer.Take(i).ToArray())
//                      .Subscribe( (fun (data:byte[]) -> if data.Length > 0 then observer.OnNext(data)
//                                                        else observer.OnCompleted()), observer.OnError, observer.OnCompleted ))
////                                                        
//module StreamExt =
//   type Stream with
//      member x.ToObservable(size) =
//         Observable.Create (fun (observer: IObserver<_>) ->
//            let buffer = Array.zeroCreate size
//            Observable.Defer(fun () -> (x.ReadAsync (buffer, 0, size)).ToObservable())
//            |> Observable.repeat
//            |> Observable.map(fun i -> buffer |> Seq.take i |> Seq.toArray)
//            |> Observable.subscribe(function
//                                    | data when data.Length > 0 -> observer.OnNext(data)
//                                    | _ -> observer.OnCompleted()) observer.OnError observer.OnCompleted)


let rec asyncFib = function
    | 0 | 1 as n -> async { return n }
    | n -> async {  let! f = asyncFib(n - 2) |> Async.StartChild
                    let! n = asyncFib(n - 1)
                    let! m =f
                    return m + n }


let parallelWorker n f =
    MailboxProcessor.Start(fun inbox ->
                            let workers = Array.init n (fun _ -> MailboxProcessor.Start(f))
                            let rec loop i = async {
                                let! msg = inbox.Receive()
                                workers.[i].Post(msg)
                                return! loop((i+1)%n)
                            }
                            loop 0)

(*(fun inbox ->
                                let rec loop() = async {
                                    let! (msg:msgTest) = inbox.Receive()
                                    match msg with
                                    | Data n -> do! Async.Sleep( 100 * n)                                                
                                    | Fetch (n, replay) -> let res = n * n
                                                           do! Async.Sleep( 100 * n )
                                                           replay.Reply(res)
                                    printfn "Thread id %d" System.Threading.Thread.CurrentThread.ManagedThreadId
                                    return! loop()
                                }*)


let agent f = parallelWorker (Environment.ProcessorCount) f

let forkJoinParallel (taskSeq) =
    Async.FromContinuations(fun (cont, econt, ccont) ->
        let tasks = Seq.toArray taskSeq
        let count = ref tasks.Length
        let results = Array.zeroCreate tasks.Length
        tasks |> Array.iteri(fun i p ->
            Async.Start( async{
                let! res = p
                results.[i] <-res
                let n = System.Threading.Interlocked.Decrement(count)
                if n=0 then cont results
            })))

let rec fib n = if n <= 2 then 1 else fib(n-1) + fib(n-2)
let fibs =  Async.Parallel[for i in 0..40 -> async{ return fib i } ]
            |> Async.RunSynchronously
fibs |> Array.iter ( printfn "%d" )// (fun i -> printfn "%d" i)