namespace Easj360FSharp

module fracture = 

    open System
    open System.Threading
    open System.Reflection
    
    [<Interface>]
    type IPipeletInput<'a> =
        abstract Post: 'a -> unit

    type Message<'a> =
    | Payload of 'a

    type pipelet<'a,'b>(name:string, transform, router: seq<IPipeletInput<'b>> * 'b -> seq<IPipeletInput<'b>>, maxcount, maxwait:int)= 
        
        let routes = ref List.empty<IPipeletInput<'b>>
        let ss  = new SemaphoreSlim(maxcount, maxcount);
        
        let dorouting result routes=
                do result |> Seq.iter (fun msg -> router(routes, msg) |> Seq.iter (fun stage -> stage.Post(msg)))

        let mailbox = MailboxProcessor.Start(fun inbox ->
            let rec loop() = async {
                let! msg = inbox.Receive()
                ss.Release() |> ignore
                try
                    match !routes with
                    | [] ->()
                    | _ as routes -> 
                        msg |> transform |> dorouting <| routes
                    return! loop()
                with //force loop resume on error
                | ex -> 
                    Console.WriteLine(sprintf "%A Error: %A" DateTime.Now.TimeOfDay ex.Message )
                    return! loop()
                }
            loop())

        interface IPipeletInput<'a> with
            member this.Post payload =
                if ss.Wait(maxwait) then
                    mailbox.Post(payload)
                else ( Console.WriteLine(sprintf "%A Overflow: %A" DateTime.Now.TimeOfDay payload ))  //overflow

        member this.Name with get() = name

        member this.Attach(stage) =
            let current = !routes
            routes := stage :: current
        
        member this.Detach (stage) =
            let current = !routes
            routes := List.filter (fun el -> el <> stage) current

    let inline (<--) (m:pipelet<_,_>) msg = (m :> IPipeletInput<_>).Post(msg)
    let inline (-->) msg (m:pipelet<_,_>)= (m :> IPipeletInput<_>).Post(msg)

    let inline (++>) (a:pipelet<_,_>) b = a.Attach(b);b
    let inline (-+>) (a:pipelet<_,_>) b = a.Detach(b);b

    /////

//    let reverse (s:string) = 
//        new string(s |> Seq.toArray |> Array.rev)
//
//    let oneToSingleton a b f=
//            let result = b |> f 
//            result |> Seq.singleton
//
//    /// total number to run through test cycle
//    let number = 100000
//
//    /// hack to record when we are done
//    let counter = ref 0
//    let sw = new Stopwatch()
//    let countthis (a:String) =
//        do Interlocked.Increment(counter) |> ignore
//        if !counter % number = 0 then 
//            sw.Stop()
//            printfn "Execution time: %A" sw.Elapsed
//            printfn "Items input: %d" number
//            printfn "Time per item: %A ms (Elapsed Time / Number of items)" (TimeSpan.FromTicks(sw.ElapsedTicks / int64 number).TotalMilliseconds)
//            printfn "Press a key to exit."
//        counter|> Seq.singleton
//
//    let OneToSeqRev a b = 
//        //Console.WriteLine(sprintf "stage: %s item: %s" a b)
//        oneToSingleton a b reverse 
//
//    /// Simply picks the first route
//    let basicRouter( r, i) =
//        r|> Seq.head |> Seq.singleton
//
//    let generateCircularSeq (s) = 
//        let rec next () = 
//            seq {
//                for element in s do
//                    yield element
//                yield! next()
//            }
//        next()
//             
//    let stage1 = pipelet("Stage1", OneToSeqRev "1", basicRouter, number, -1)
//    let stage2 = pipelet("Stage2", OneToSeqRev "2", basicRouter, number, -1)
//    let stage3 = pipelet("Stage3", OneToSeqRev "3", basicRouter, number, -1)
//    let stage4 = pipelet("Stage4", OneToSeqRev "4", basicRouter, number, -1)
//    let stage5 = pipelet("Stage5", OneToSeqRev "5", basicRouter, number, -1)
//    let stage6 = pipelet("Stage6", OneToSeqRev "6", basicRouter, number, -1)
//    let stage7 = pipelet("Stage7", OneToSeqRev "7", basicRouter, number, -1)
//    let stage8 = pipelet("Stage8", OneToSeqRev "8", basicRouter, number, -1)
//    let stage9 = pipelet("Stage9", OneToSeqRev "9", basicRouter, number, -1)
//    let stage10 = pipelet("Stage10", OneToSeqRev "10", basicRouter, number, -1)
//    let final = pipelet("Final", countthis, basicRouter, number, -1)
//
//    stage1 
//    ++> stage2
//    ++> stage3
//    ++> stage4 
//    ++> stage4 
//    ++> stage5 
//    ++> stage6 
//    ++> stage7 
//    ++> stage8 
//    ++> stage9 
//    ++> stage10 
//    ++> final 
//    ++> {new IPipeletInput<_> with member this.Post payload = () }|> ignore
//
//    //remove stage2 from stage1
//    //stage1 -+> stage2 |> ignore
//      
//    System.AppDomain.CurrentDomain.UnhandledException |> Observable.add (fun x -> 
//        printfn "%A" (x.ExceptionObject :?> Exception);Console.ReadKey() |>ignore)
//
//    sw.Start()
//    for str in ["John"; "Paul"; "George"; "Ringo"; "Nord"; "Bert"] 
//    |> generateCircularSeq 
//    |> Seq.take number
//        do  str --> stage1
//
//    Console.WriteLine("Insert complete waiting for operation to complete.")
//    let x = Console.ReadKey()