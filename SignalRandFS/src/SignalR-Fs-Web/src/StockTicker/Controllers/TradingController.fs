namespace StockTicker.Controllers

open System
open System.Net
open System.Net.Http
open System.Web.Http
open StockTicker.Rop
open StockTicker
open StockTicker.Validation
open StockTicker.Commands
open System.Reactive.Subjects
open StockTickerServer


[<RoutePrefix("api/trading")>]
type TradingController() = 
    inherit ApiController() 

    // the controller act as a observable publisher of messages 
   
    // keep the controller loosely coupled using the Reactive Extensions
    // defining a Reactive Subject which will fire up
    // requests to the subscriber(s)
    let subject = new Subject<CommandWrapper>()
    
    let log res =
        let msg = 
            match res with
            | Success(m) -> sprintf "Validation successful - %A" m
            | Failure(f) -> sprintf "Validation failled - %s" f
        System.Diagnostics.Debug.WriteLine("[LOG]" + msg)
        res

    let publish connectionId cmd =
            match cmd with
            | Result.Success(c) -> 
                let commandWarepper = c |> CommandWrapper.CreateTradingCommand connectionId
                subject.OnNext commandWarepper
            | Result.Failure(e) -> subject.OnError(exn (e))
            cmd

    let toResponse (request : HttpRequestMessage) result = 
        let response = 
            match result with
            | Success(_) -> request.CreateResponse(HttpStatusCode.OK)
            | _ -> request.CreateResponse(HttpStatusCode.BadRequest)
        response
    
    [<Route("sell"); HttpPost>]
    member this.PostSell([<FromBody>] tr : TradingRequest) = 
        async { 
            // current connection ID from SignalR
            let connectionId = tr.ConnectionID 
            
            // create TradingCommand
            // validate
            // log
            // publish
            // return response
            return // TradingCommand
                {   Symbol = tr.Symbol.ToUpper()  
                    Quantity = tr.Quantity
                    Price = tr.Price
                    Trading = TradingType.Sell }
                |> tradingdValidation // validation using function composition
                |> log
                |> publish connectionId 
                |> toResponse this.Request
         
        } |> Async.StartAsTask // can easily make asynchronous controller methods.
    
    [<Route("buy"); HttpPost>]  
    member this.PostBuy([<FromBody>] tr : TradingRequest) = 
        async { 
            let connectionId = tr.ConnectionID
            
            return
                { Symbol = (tr.Symbol.ToUpper())
                  Quantity = tr.Quantity
                  Price = tr.Price
                  Trading = TradingType.Buy }
                |> tradingdValidation
                |> log
                |> publish connectionId 
                |> toResponse this.Request

        } |> Async.StartAsTask // can easily make asynchronous controller methods.
    
    [<Route("addTicker"); HttpPost>]  
    member this.PostAddTicker([<FromBody>] tr : TradingRequest) = 
        async { 
            let connectionId = tr.ConnectionID
            
            let commandTickerRequest =
                {   TickerRecord.Symbol = (tr.Symbol.ToUpper())
                    TickerRecord.Price = tr.Price }
                |> tickerRequestValidation
                |> log
            match commandTickerRequest with
            | Result.Success(c) -> 
                let commandWarepper = c |> CommandWrapper.CreateTickerRequestCommand connectionId
                subject.OnNext commandWarepper
            | Result.Failure(e) -> subject.OnError(exn (e))
                        
            return commandTickerRequest |> (toResponse this.Request)
        } |> Async.StartAsTask // can easily make asynchronous controller methods.
    
    [<Route("buyTicker") ; HttpPost>]  
    member this.PostBuyTicker([<FromBody>] tr : TradingRequest) : HttpResponseMessage = 
            match base.ModelState.IsValid with
            | true ->   let connectionId = tr.ConnectionID
            
                        let command = 
                            {   Symbol = (tr.Symbol.ToUpper())
                                Quantity = tr.Quantity
                                Price = tr.Price
                                Trading = TradingType.Sell }
                            |> tradingdValidation
                            |> log
                        match command with
                        | Success(c) -> 
                            let commandWarepper = 
                                c |> CommandWrapper.CreateTradingCommand connectionId
                            subject.OnNext commandWarepper
                        | Failure(e) -> subject.OnError(exn (e))
                        command |> (toResponse this.Request)
            | false -> this.Request.CreateErrorResponse(HttpStatusCode.BadRequest, "Model Error")

    // The controller bahaves as Observable publisher and it can be register
    interface IObservable<CommandWrapper> with
        member this.Subscribe observer = subject.Subscribe observer
    
    override this.Dispose disposing = 
        if disposing then 
            subject.Dispose()
        base.Dispose disposing

