
[<AutoOpenAttribute>]
module TradingSupervisorAgent

 
open System
open System.Collections.Generic
open System.Threading
open Microsoft.AspNet.SignalR
open Microsoft.AspNet.SignalR.Hubs
open System.Reactive.Subjects
open System.Reactive.Linq
open StockTickerServer
open TradingAgent 
open StockTickerHubClient
        
type TradingSupervisorMessage =
    | Subscribe of  id : string * initialAmount : float *  caller:IHubCallerConnectionContext<IStockTickerHubClient>
    | Unsubscribe of   id : string
    | Buy of id : string * Ticker : string * TradingDetails
    | Sell of id : string * Ticker : string * TradingDetails
    | AddStock of connId:string * stock:Stock

   
 
// responsible for subscribing and unsubscribing TradingAgent
// it uses a mix of RX and Agent.Post just for demo purpose
// (TradingAgent : IOboservable) and (TradingSuperviserAgent : IObservable)
type internal TradingSupervisorAgent() =                
    
    let subject = new Subject<TradingMessage>()
 
    let agent =
        Agent<TradingSupervisorMessage>.Start(fun inbox ->
                
            let rec loop n (agents : Map<string, (IObserver<TradingMessage> * IDisposable)>) =
                async {
                    let! msg = inbox.Receive()
                  
                    match msg with                           
                    | Subscribe(id, initialAmount, caller) ->
                        let agent = Map.tryFind id agents
                          
                        match agent with
                        | Some(a) -> return! loop n agents
                        | None ->
                            let observerAgent = new TradingAgent(id, initialAmount, caller)
                               
                            let dispObsrever = subject.Subscribe(observerAgent)
                            observerAgent.Agent |> reportErrorsTo id supervisor |> startAgent

                            caller.Client(id).SetInitialAsset(initialAmount)
                            return! loop (n + 1) (Map.add id (observerAgent :> IObserver<TradingMessage>, dispObsrever) agents)
 
                    | Unsubscribe(id) ->
                        let agent = Map.tryFind id agents
                        match agent with
                        | Some(a, d) ->    a.OnCompleted()
                                           d.Dispose() // Agent dispsed 
                                           return! loop (n - 1) (Map.remove id agents)
                        | None -> return! loop n agents
                        
                    | Buy(id, symbol, trading) ->
                        let agent = Map.tryFind id agents
                        match agent with
                        | Some(a, _) ->
                            a.OnNext(TradingMessage.Buy(symbol, trading))
                            return! loop n agents
                        | None -> return! loop n agents
                        
                    | Sell(id, symbol, trading) ->
                        let agent = Map.tryFind id agents
                        match agent with
                        | Some(a, _) ->
                            a.OnNext(TradingMessage.Sell(symbol, trading))
                            return! loop n agents
                        | None -> return! loop n agents
 
                        
                    | AddStock(id, stock) ->
                        let agent = Map.tryFind id agents
                        match agent with
                        | Some(a, _) ->
                            a.OnNext(TradingMessage.AddStock(stock))
                            return! loop n agents
                        | None -> return! loop n agents
                        }
            loop 0 (Map.empty))

    // exposes and IObservable
    let obs = subject :> IObservable<TradingMessage> 
 
    member this.OnUpdateStock(stock:Stock) =
        // in this case the OnNext is bypassing the local Agent
        // and all the clients will be notify
        subject.OnNext(TradingMessage.UpdateStock(stock))
        
    member this.OnNotify(msg:TradingSupervisorMessage) =
            agent.Post(msg)
       
    member x.Subscribe(id : string, initialAmount : float, caller:IHubCallerConnectionContext<IStockTickerHubClient>) =
        agent.Post(Subscribe(id, initialAmount, caller))
      
    member x.Unsubscribe(id : string) = agent.Post(Unsubscribe(id))
       
    interface IObservable<TradingMessage> with
        member x.Subscribe(obs) =
            subject.Subscribe(obs)
