open System

let flip f x y = f y x

let rec cycle s =
    seq {
        yield! s
        yield! cycle s
    }

type Agent<'T> = MailboxProcessor<'T>

type Message =
    | Waiting of (Set<int> * AsyncReplyChannel<unit>)
    | Done of Set<int>

let reply (c : AsyncReplyChannel<_>) = c.Reply()

let strategy forks waiting =
    let aux, waiting = List.partition (fst >> flip Set.isSubset forks) waiting

    let forks =
        aux
        |> List.map fst
        |> List.fold (-) forks
    List.iter (snd >> reply) aux
    forks, waiting

let waiter strategy forkCount =
    Agent<_>.Start(fun inbox ->
        let rec loop forks waiting =
            async {
                let forks, waiting = strategy forks waiting
                let! msg = inbox.Receive()
                match msg with
                | Waiting r -> return! loop forks (waiting @ [ r ])
                | Done f -> return! loop (forks + f) waiting
            }
        loop (Set.ofList (List.init forkCount id)) [])

let philosopher (waiter : Agent<_>) name forks =
    let rng = new Random()
    let forks = Set.ofArray forks
    Agent<_>.Start(fun inbox ->
        let rec loop() =
            async {
                printfn "%s is thinking" name
                do! Async.Sleep(rng.Next(100, 500))
                printfn "%s is hungry" name
                do! waiter.PostAndAsyncReply(fun c -> Waiting(forks, c))
                printfn "%s is eating" name
                do! Async.Sleep(rng.Next(100, 500))
                printfn "%s is done eating" name
                waiter.Post(Done(forks))
                return! loop()
            }
        loop())

let test() =
    let forks =
        Seq.init 5 id
        |> cycle
        |> Seq.windowed 2
        |> Seq.take 5
        |> Seq.toList

    let names = [ "plato"; "aristotel"; "kant"; "nietzsche"; "russel" ]
    let waiter = waiter strategy 5
    List.map2 (philosopher waiter) names forks |> ignore

//////////// FIBONACCI TESTS ///////
let rec fib =
    function
    | 0 | 1 as n -> n
    | n -> fib (n - 1) + fib (n - 2)

fib 25

let rec fibAsync =
    function
    | 0 | 1 as n -> async { return n }
    | n -> async { let! m = fibAsync (n - 1)
                   let! n = fibAsync (n - 2)
                   return m + n }

Async.RunSynchronously(fibAsync 25)

let rec fibChild =
    function
    | 0 | 1 as n -> async { return n }
    | n -> async { let! f = fibChild (n - 2) |> Async.StartChild
                   let! n = fibChild (n - 1)
                   let! m = f
                   return m + n }

Async.RunSynchronously(fibChild 25)

module BadFib =
    let rec parfib =
        function
        | 0 | 1 as n -> n
        | n ->
            let p = System.Threading.Tasks.Task.Factory.StartNew(fun () -> parfib (n - 2))
            let q = parfib (n - 1)
            p.Result + q

module GoodFib =
    let rec parfib i =
        function
        | 0 | 1 as n -> n
        | n when n <= i -> parfib i (n - 2) + parfib i (n - 1)
        | n ->
            let p = System.Threading.Tasks.Task.Factory.StartNew(fun () -> parfib i (n - 2))
            let q = parfib i (n - 1)
            p.Result + q

BadFib.parfib 30
GoodFib.parfib 30

