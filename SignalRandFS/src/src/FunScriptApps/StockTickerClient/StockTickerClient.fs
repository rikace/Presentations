[<ReflectedDefinition>]
module StockTickerSignalRClient 

open FunScript.TypeScript
open FunScript
open SignalRProvider
open System.Collections.Generic
open FunScript.TypeScript
open FunScript.HTML

// Utility function to make adding values to a dictionary more F#esque
let add (key: _) (value: obj) (dic: Dictionary<_,_>) =
    dic.Add(key, unbox value); dic
    
let signalR = Globals.Dollar.signalR
let j (s: string) = Globals.Dollar.Invoke(s)
let log (msg:string) = Globals.console.log msg

let serverHub = new Hubs.stockTicker(signalR.hub)

let jqIgnore x = 
    x
    null : obj

type OrderRecord = 
    {   Symbol : string
        Quantity : int
        Price : float
        OrderPrice : float
        OrderType:string }

type Asset = {Cash:float; Portfolio:List<OrderRecord>; BuyOrders:List<OrderRecord>; SellOrders:List<OrderRecord>}

type Stock = 
    { Symbol : string
      DayOpen : string
      DayLow : string
      DayHigh : string
      LastChange : string
      Price : string 
      PercentChange : string
      Direction : string
      DirectionClass : string
      Index : int}

type TikcerRequest = {ConnectionID:string; Symbol:string; Price:float}

type OrderRequest = {ConnectionID:string; Symbol:string; Price:float; Quantity : int}

//type StockClient = SignalRProvider.Types.``StockTickerServer!Models+Stock``

type System.Net.WebRequest with
    member this.AsyncPostJSONOneWay<'T>(url:string, data: 'T) =
        let req: FunScript.Core.Web.WebRequest = unbox this
        req.Headers.Add("Accept", "application/json")
        req.Headers.Add("Content-Type", "application/json")
        let onReceived(data:string) = () 
        let onErrorReceived() = ()
        async {
            ignore <| FunScript.Core.Web.sendRequest(
                "POST",url , req.Headers.Keys, req.Headers.Values, 
                Globals.JSON.stringify(data), (fun d -> ()), (fun () ->())) }

let onstart () = 
    serverHub.GetAllStocks() |> ignore
    log "##Started!##"


[<JSEmit("{0}.animate({ backgroundColor: 'rgb(' + {1} + ')' }, 1000 / 2);")>]
let animateColorIn(x:JQuery, color:string) : unit = failwith "never"

[<JSEmit("{0}.animate({ backgroundColor: {1} }, 1000 / 2);")>]
let animateColorOut(x:JQuery, color:string) : unit = failwith "never"

[<JSEmitInlineAttribute("({0} * 1.0)")>]
let number(x : int) : float = failwith "never"

[<JSEmitInlineAttribute("(new Date()).getTime()")>]
let getTime() : int = failwith "never"

let stopScrollTicker() = j("#stockTicker").find("ul").stop()
   
let init() =    
    let data = 
        Dictionary<string, obj>()
        |> add "stocks" (ResizeArray<Stock>())        
        |> add "asset" (ResizeArray<OrderRecord>())
        |> add "buyOrders" (ResizeArray<OrderRecord>())
        |> add "sellOrders" (ResizeArray<OrderRecord>())

    let options = createEmpty<RactiveNewOptions>()
    options.el <- "#stockStickerApp"
    options.template <- "#main"    
    options.data <- data
    options.twoway <- false // disable two way data binding
    let ractive = Globals.Ractive.Create(options)
    (ractive, data)

let stockTickerProcess ractive stocks =
    let rec waitingLoop(r: Ractive, stocks: List<Stock>): Async<unit> = async {
        let ev1, ev2, ev3 = r.onStream("open", "close", "buyStock")        
        let! choice = Async.AwaitObservable(ev1, ev2, ev3)
        match choice with
        | Choice1Of3 (ev, arg) ->           
            serverHub.OpenMarket() |> ignore
            j("#open").prop("disabled", true) |> ignore
            j("#close").prop("disabled", false) |> ignore            
        | Choice2Of3 (ev, arg) ->
            serverHub.CloseMarket() |> ignore            
            j("#open").prop("disabled", false) |> ignore
            j("#close").prop("disabled", true) |> ignore
            stopScrollTicker() |> ignore
        | Choice3Of3 (ev, arg) -> 
            let tickerSymbolBuy = j("#tickerSymbolBuy")
            let tickerPriceBuy = j("#tickerPriceBuy")
            let tickerQuantityBuy = j("#tickerQuantityBuy")
            let symbol:string = tickerSymbolBuy._val() |> unbox
            let priceStr:string = tickerPriceBuy._val() |> unbox
            let quantity:string = tickerQuantityBuy._val() |> unbox
            tickerSymbolBuy._val("") |> ignore
            tickerPriceBuy._val("") |> ignore    
            tickerQuantityBuy._val("") |> ignore    
            let orderRequestBuy = {ConnectionID = signalR.hub.id; Symbol = symbol; Quantity = int(quantity);  Price = (float priceStr)}
            async{  let url = "http://localhost:48430/api/trading/Buy"
                    let req = System.Net.WebRequest.Create(url)
                    do! req.AsyncPostJSONOneWay(url, orderRequestBuy)
                     } |> Async.StartImmediate
        return! waitingLoop(r, stocks) // Repeat the loop 
    }
    Async.StartImmediate <| waitingLoop(ractive, stocks)

let orderProcess (ractive:Ractive) =
    ractive.onStream("addTicker")
    |> Observable.add (fun (ev, arg) ->
        let tickerSymbol = j("#tickerSymbol")
        let tickerPrice = j("#tickerPrice")
        let symbol:string = tickerSymbol._val() |> unbox
        let priceStr:string = tickerPrice._val() |> unbox
        tickerSymbol._val("") |> ignore
        tickerPrice._val("") |> ignore    
        let tikcerRequest = {ConnectionID = signalR.hub.id; TikcerRequest.Symbol = symbol; Price = (float priceStr)}
        async{  let url = "http://localhost:48430/api/trading/addTicker"
                let req = System.Net.WebRequest.Create(url)
                do! req.AsyncPostJSONOneWay(url, tikcerRequest)
            } |> Async.StartImmediate)

    ractive.onStream("sellStock")
    |> Observable.add (fun (ev, arg) ->
        let tickerSymbolSell = j("#tickerSymbolSell")
        let tickerPriceSell = j("#tickerPriceSell")
        let tickerQuantitySell = j("#tickerQuantitySell")

        let symbol:string = tickerSymbolSell._val() |> unbox
        let priceStr:string = tickerPriceSell._val() |> unbox
        let quantity:string = tickerQuantitySell._val() |> unbox
        tickerSymbolSell._val("") |> ignore
        tickerPriceSell._val("") |> ignore    
        tickerQuantitySell._val("") |> ignore    
        let orderRequestSell = {ConnectionID = signalR.hub.id; Symbol = symbol; Quantity = int(quantity);  Price = (float priceStr)}
        async{  let url = "http://localhost:48430/api/trading/Sell"
                let req = System.Net.WebRequest.Create(url)
                do! req.AsyncPostJSONOneWay(url, orderRequestSell)
            } |> Async.StartImmediate)

let main() = 
    Globals.console.log("##Starting:## ")
    signalR.hub.url <- "http://localhost:48430/signalr/hubs"

    let client = Hubs.StockTickerHubClient()

    let reactive, data = init()

    client.SetInitialAsset <- (fun amount -> 
                                    let cash = (sprintf "%.2f" amount)
                                    j("#portfolioCash")._val(cash) |> ignore )

    client.UpdateOrderBuy <- (fun order ->
                                    let o:OrderRecord = unbox (Globals.jQuery.parseJSON((sprintf "%A" order)))
                                    let buyOrders:List<OrderRecord> = (unbox data.["buyOrders"])
                                    buyOrders.Add(o))
    
    client.UpdateOrderSell <- (fun order ->
                                    let o:OrderRecord = unbox (Globals.jQuery.parseJSON((sprintf "%A" order)))
                                    let sellOrders:List<OrderRecord> = (unbox data.["sellOrders"])
                                    sellOrders.Add(o))

    client.SetStock <- (fun stock ->
                 let s:Stock = unbox (Globals.jQuery.parseJSON((sprintf "%A" stock)))            
                 let stocks = ((unbox data.["stocks"]) :> List<Stock>)
                 if not <| (stocks |> Seq.exists(fun t -> t.Symbol.ToUpper() = s.Symbol.ToUpper())) then 
                    stocks.Add(s) )
    
    client.UpdateAsset <- (fun newAsset -> 
                            
                            let asset:Asset =  unbox (Globals.jQuery.parseJSON((sprintf "%A" newAsset))) 
                            let buyOrders:List<OrderRecord> = (unbox data.["buyOrders"])
                            let portfolio:List<OrderRecord> = (unbox data.["asset"]) 
                            let sellOrders:List<OrderRecord> = (unbox data.["sellOrders"])

                            let cash = (sprintf "%.2f" asset.Cash)
                            j("#portfolioCash")._val(cash) |> ignore

                            if asset.Portfolio |> Seq.length > 0 then
                                if portfolio |> Seq.length = 0 then
                                    for p in asset.Portfolio do
                                        let price = float (sprintf "%.2f" p.Price)
                                        portfolio.Add({p with OrderPrice = price ;Price = price })
                                else
                                    for p in asset.Portfolio do                                       
                                        if portfolio |> Seq.exists(fun f -> f.Symbol = p.Symbol) then
                                            let index = portfolio |> Seq.findIndex(fun f -> f.Symbol = p.Symbol)
                                            portfolio.[index] <- {p with OrderPrice = portfolio.[index].OrderPrice ; Price = float (sprintf "%.2f" p.Price)}
                                        else
                                            let price = float (sprintf "%.2f" p.Price)
                                            portfolio.Add({p with OrderPrice = price ;Price = price })                                            
                            else
                                for i = 0 to ((portfolio |> Seq.length) - 1) do
                                    portfolio.RemoveAt(0)

                            if asset.BuyOrders |> Seq.length > 0 then
                                if buyOrders |> Seq.length = 0 then
                                    for o in asset.BuyOrders do
                                        buyOrders.Add(o)
                                else
                                    for o in asset.BuyOrders do
                                        if buyOrders |> Seq.exists(fun f -> f.Symbol = o.Symbol && f.Price = o.Price) then
                                            let index = buyOrders |> Seq.findIndex(fun f -> f.Symbol = o.Symbol && f.Price = o.Price)
                                            buyOrders.[index] <- o
                                        else
                                            buyOrders.Add(o)
                            else
                                for i = 0 to ((buyOrders |> Seq.length) - 1) do
                                    buyOrders.RemoveAt(0)

                            for o in buyOrders do
                                if not <| (asset.BuyOrders |> Seq.exists(fun f -> f.Symbol = o.Symbol && f.Price = o.Price)) then
                                    buyOrders.Remove(o) |> ignore
                                
                            if asset.SellOrders |> Seq.length > 0 then
                                if sellOrders |> Seq.length = 0 then
                                    for o in asset.SellOrders do
                                        sellOrders.Add(o)
                                else
                                    for o in asset.SellOrders do
                                        if sellOrders |> Seq.exists(fun f -> f.Symbol = o.Symbol && f.Price = o.Price) then
                                            let index = sellOrders |> Seq.findIndex(fun f -> f.Symbol = o.Symbol  && f.Price = o.Price)
                                            sellOrders.[index] <- o
                                        else
                                            sellOrders.Add(o)
                            else
                                for i = 0 to ((sellOrders |> Seq.length) - 1) do
                                    sellOrders.RemoveAt(0)

                            for o in sellOrders do
                                if not <| (asset.SellOrders |> Seq.exists(fun f -> f.Symbol = o.Symbol && f.Price = o.Price)) then
                                    sellOrders.Remove(o) |> ignore
                        )

    client.UpdateStockPrice <- (fun (stock) -> 
                 let s:Stock = unbox (Globals.JSON.parse((sprintf "%A" stock)))                                  
                 let s' = {     Symbol = s.Symbol
                                DayOpen = sprintf "%.2f" (float s.DayOpen)
                                DayLow = sprintf "%.2f" (float s.DayLow)
                                DayHigh = sprintf "%.2f" (float s.DayHigh)
                                LastChange = sprintf "%.2f" (float s.LastChange)
                                Price = sprintf "%.2f" (float s.Price)
                                PercentChange =sprintf "%.4f %%" ((float  s.PercentChange) * 100.)
                                Direction = if int(s.LastChange) = 0 then ""
                                            elif int(s.LastChange) >= 0 then "▲"
                                            else "▼"
                                DirectionClass = if int(s.LastChange) = 0 then "even"
                                                 elif int(s.LastChange) >= 0 then "up"
                                                 else "down"
                                Index = s.Index } 

                 ((unbox data.["stocks"]) :> List<Stock>).[s.Index] <- s')

    client.Register(signalR.hub)
    
    stockTickerProcess reactive (unbox data.["stocks"])
    orderProcess reactive

    signalR.hub.start onstart

type Wrapper() =
    member this.GenerateScript() = Compiler.compileWithoutReturn <@ main() @>

 

