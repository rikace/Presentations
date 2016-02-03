namespace AgentModule


[<AutoOpenAttribute>]
module Utils =

    open System
    open System.Reactive
    open System.Reactive.Linq
    open System.Reactive.Subjects
    open System.Threading
    open System.Threading.Tasks

    
    type Observable with
        static member buffer selector o = Observable.Buffer(o, (fun b -> selector(o)))
        static member groupBy selector o = Observable.GroupBy(o, (fun v -> selector(v)))
        static member collect (selector : 'a -> IObservable<'b>) o = Observable.SelectMany(o, selector)


    let internal synchronize f = 
        let ctx = System.Threading.SynchronizationContext.Current 
        f (fun g ->
          let nctx = System.Threading.SynchronizationContext.Current 
          if ctx <> null && ctx <> nctx then ctx.Post((fun _ -> g()), null)
          else g() )

    
    type Agent<'T> = MailboxProcessor<'T>

    [<AutoOpenAttribute>]
    module ThreadSafeRandom =

        type private ThreadSafeRandomRequest =
        | GetDouble of AsyncReplyChannel<float>

        let private agent = Agent.Start(fun inbox -> 
                let rnd = new Random()
                let rec loop() = async {
                    let! msg = inbox.Receive()
                    match msg with
                    | GetDouble(reply) -> reply.Reply(rnd.NextDouble())
                    return! loop()
                }
                loop() )

        let getThreadSafeRandom() = agent.PostAndReply(fun ch -> GetDouble(ch))



    type Stock = 
      { Symbol : string
        LastPrice : float
        Price : float }
       member x.Change = x.Price - x.LastPrice
       member x.UpdatePrice price = {x with LastPrice = x.Price; Price = price }

       static member CreateStock (symbol : string) price = 
            { Symbol = symbol
              LastPrice = price
              Price = price }
     
       member x.PercentChange = double (Math.Round(x.Change / x.Price, 4))     
 
       member x.Update() = 
        let r = ThreadSafeRandom.getThreadSafeRandom()
        if r > 0.1 then x
        else
            let rnd' = Random(int (Math.Floor(x.Price)))
            let percenatgeChange = rnd'.NextDouble() * 0.002
            let change =
                let change = Math.Round(x.Price * percenatgeChange, 2)
                if (rnd'.NextDouble() > 0.51) then change
                else -change
            let price = x.Price + change
            { x with  LastPrice = price
                      Price = price }

    
    type Task<'a> with
        member t.ToObservale() : IObservable<'a> =
            let subject = new AsyncSubject<'a>()
            t.ContinueWith(fun (f:Task<'a>) ->
            if f.IsFaulted then
                subject.OnError(f.Exception)
            else
                subject.OnNext(f.Result)
                subject.OnCompleted() ) |> ignore
            subject :> IObservable<'a>
    
    type Async<'a> with
        member a.ToObservable() =        
            let subject = new AsyncSubject<'a>()
            Async.StartWithContinuations(a,
                (fun res -> subject.OnNext(res)
                            subject.OnCompleted()),
                (fun exn -> subject.OnError(exn)),
                (fun cnl -> ()))
            subject :> IObservable<'a>
