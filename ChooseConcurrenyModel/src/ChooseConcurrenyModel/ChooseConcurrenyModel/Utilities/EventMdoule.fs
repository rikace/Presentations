namespace EventModule

[<AutoOpenAttribute>]
module Event =
    let split3 (f: 'T -> Choice<'U1, 'U2, 'U3>) (e:IEvent<'Del, 'T>) =
        let ev1 = new Event<_>()
        let ev2 = new Event<_>()
        let ev3 = new Event<_>()
        e.Add(f >> function
            | Choice1Of3 x -> ev1.Trigger(x)
            | Choice2Of3 x -> ev2.Trigger(x)
            | Choice3Of3 x -> ev3.Trigger(x))
        (ev1.Publish, ev2.Publish, ev3.Publish)


//    open System.Net
//    let client = new WebClient()
//    let success, failure, cancel = 
//        client.DownloadStringCompleted 
//        |> split3 (fun args -> 
//            if args.Error <> null then Choice2Of3(args.Error)
//            elif args.Cancelled then Choice3Of3()
//            else Choice1Of3(args.Result))
    //success.Add(fun -> ())
