namespace Easj360FSharp

open System
open AgentHelper

module AsyncObjectPool =

    //type Agent<'T> = MailboxProcessor<'T>

    ///One of three messages for our Object Pool agent
    type PoolMessage<'a> =
        | Get of AsyncReplyChannel<'a>
        | Put of 'a * AsyncReplyChannel<unit>
        | Clear of AsyncReplyChannel<List<'a>>

    /// Object pool representing a reusable pool of objects
    type ObjectPool<'a>(generate: unit -> 'a, initialPoolCount) = 
        let initial = List.init initialPoolCount (fun (x) -> generate())
        let agent = Agent.Start(fun inbox ->
            let rec loop(x) = async {
                let! msg = inbox.Receive()
                match msg with
                | Get(reply) -> 
                    let res = match x with
                              | a :: b -> 
                                  reply.Reply(a);b
                              | [] as empty-> 
                                  reply.Reply(generate());empty
                    return! loop(res)
                | Put(value, reply)-> 
                    reply.Reply()
                    return! loop(value :: x) 
                | Clear(reply) -> 
                    reply.Reply(x)
                    return! loop(List.empty<'a> )            
            }
            loop(initial))

        /// Clears the object pool, returning all of the data that was in the pool.
        member this.ToListAndClear() = 
            agent.PostAndAsyncReply(Clear)
        /// Puts an item into the pool
        member this.Put(item) = 
            agent.PostAndAsyncReply((fun ch -> Put(item, ch)))
        /// Gets an item from the pool or if there are none present use the generator
        member this.Get(item) = 
            agent.PostAndAsyncReply(Get)


    type Customer = 
        {First : string; Last : string; AccountNumber : int;}
        override m.ToString() = sprintf "%s %s, Acc: %d" m.First  m.Last m.AccountNumber

    let names = ["John"; "Paul"; "George"; "Ringo"]
    let lastnames = ["Lennon";"McCartney";"Harison";"Starr";]
    let rand = System.Random()

    let randomFromList list= 
        let length = List.length list
        let skip = rand.Next(0, length)
        list |> List.toSeq |> (Seq.skip skip ) |> Seq.head

    let customerGenerator() =
        Async.RunSynchronously(Async.Sleep(100))
        { First = names |> randomFromList; 
          Last= lastnames |> randomFromList; 
          AccountNumber = rand.Next(100000, 999999);}
  
    let numberToGenerate = 10    

    let objectPool = ObjectPool(customerGenerator, numberToGenerate)

    printfn "%d customers in pool" numberToGenerate

    let numberToRun = seq { 0 .. numberToGenerate * 2 - 1 }

    let sw = System.Diagnostics.Stopwatch.StartNew()

    for x in numberToRun do 
        (   sw.Start()   
            let result = Async.RunSynchronously( async{return! objectPool.Get()})
            do sw.Stop()
            printfn "*%d %O, Generation time %A ms" x result sw.Elapsed.TotalMilliseconds
            sw.Reset()
        )

    printfn "Press any key to exit."
    Console.ReadKey() |> ignore