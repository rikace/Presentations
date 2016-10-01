namespace ChanbelAgent

module Channel =

    open System.Collections.Generic
    open System.Collections.Concurrent
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
        


    type 'a ChannelMsg =
        | Read of ('a -> unit)
        | Write of 'a * (unit -> unit)



    
    type [<Sealed>] Channel<'T>() =
        let channelAgent =
            MailboxProcessor<ChannelMsg<'T>>.Start(fun inbox ->                
                let readers = Queue()
                let writers = Queue()
                let rec loop () = async {
                    let! msg = inbox.Receive()

                    match msg with
                    | Read ok when writers.Count = 0 -> 
                        readers.Enqueue ok
                        return! loop()
                    | Read ok -> let value, cont = writers.Dequeue()                              
                                 Pool.Spawn cont
                                 ok value
                                 return! loop()
                    | Write (x,ok) when readers.Count = 0 ->
                              writers.Enqueue(x, ok)                    
                              return! loop()
                    | Write (x,ok) ->  let cont = readers.Dequeue()
                                       Pool.Spawn ok
                                       cont x
                                       return! loop() }
                loop() )

        member this.Read(read) = channelAgent.Post (Read read)
        member this.Write(v, ok) = channelAgent.Post (Write (v,ok))
        member inline this.Read() =
            Async.FromContinuations(fun (ok, _, _) -> this.Read ok)
        member inline this.Write x =
            Async.FromContinuations(fun (ok, _, _) -> this.Write(x,ok))




//     
//        member inline this.Read() =
//            Async.FromContinuations(fun (ok, _, _) ->
//                this.Read ok)
//
//        member inline this.Write x =
//            Async.FromContinuations(fun (ok, _, _) ->
//                this.Write(x, ok))

module test = 
    open Channel


    [<Sealed>]
    type Agent<'T,'R>(f: Channel<'T> -> Async<'R>, opts) =
        let chan = Channel<'T>()
        let task = Async.StartAsTask(f chan, opts)
        member this.Write(m) = chan.Write(m)
        member this.Read() = chan.Read()


    

    let n = 100000


    let chan = Channel<int>()

    let rec writer (i: int) =
        async {
            if i = 0 then
                return! chan.Write 0
            else
                do! chan.Write i
                return! writer (i - 1)
        }

    let rec reader(sum:int) =
        async {
            let! x = chan.Read()
            if x = 0 then 
                return sum
            else 
                return! reader (sum + x)
        }

    Async.Start(writer n)
    let clock = System.Diagnostics.Stopwatch()
    clock.Start()
    
    let r = Async.RunSynchronously(reader 0)
    stdout.WriteLine("Hops per second: {0}", float n / clock.Elapsed.TotalSeconds)
    

