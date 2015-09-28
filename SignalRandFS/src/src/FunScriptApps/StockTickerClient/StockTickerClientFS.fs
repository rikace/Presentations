[<ReflectedDefinition>]
module StockTickerClientFSClient

open FunScript.TypeScript
open FunScript
open SignalRProvider
open System.Collections.Generic
open FunScript.TypeScript
open FunScript.HTML

// Utility function to make adding values to a dictionary more F#esque
let add (key : _) (value : obj) (dic : Dictionary<_, _>) = 
    dic.Add(key, unbox value)
    dic

let signalR = Globals.Dollar.signalR

let j (s : string) = Globals.Dollar.Invoke(s)

let log (msg : string) = Globals.console.log msg

// check type of .stockTicker
let serverHub = new Hubs.stockTicker(signalR.hub)

// types used by Stock-Ticker
type OrderRecord = SignalRProvider.Types.``StockTickerServer!Models+OrderRecord``

type Asset = SignalRProvider.Types.``StockTickerServer!Models+Asset``

type Stock = SignalRProvider.Types.``StockTickerServer!Models+Stock``

type TikcerRequest =
    { ConnectionID : string
      Symbol : string
      Price : float 
      Quantity : int }


// I can create operation leveraging F# Async Workflow !!!!!
// and it will compile in JS
type System.Net.WebRequest with
    member this.AsyncPostJSONOneWay<'T>(url : string, data : 'T) = 
        let req : FunScript.Core.Web.WebRequest = unbox this
        req.Headers.Add("Accept", "application/json")
        req.Headers.Add("Content-Type", "application/json")
       
        let onReceived (data : string) = ()
        let onErrorReceived() = ()
        
        async { ignore <| FunScript.Core.Web.sendRequest 
                   ("POST", url, req.Headers.Keys, req.Headers.Values, Globals.JSON.stringify (data), 
                    onReceived, 
                    onErrorReceived) }

let onstart() = 
    serverHub.GetAllStocks() |> ignore
    log "##Started!##"

[<JSEmit("{0}.animate({ backgroundColor: 'rgb(' + {1} + ')' }, 1000 / 2);")>]
let animateColorIn (x : JQuery, color : string) : unit = failwith "never"

[<JSEmit("{0}.animate({ backgroundColor: {1} }, 1000 / 2);")>]
let animateColorOut (x : JQuery, color : string) : unit = failwith "never"

[<JSEmitInlineAttribute("({0} * 1.0)")>]
let number (x : int) : float = failwith "never"

[<JSEmitInlineAttribute("(new Date()).getTime()")>]
let getTime() : int = failwith "never"


let stopScrollTicker() = j("#stockTicker").find("ul").stop()

let startScrollTicker() = j("#stockTicker").find("ul").scroll()

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
    options.twoway <- false
    let ractive = Globals.Ractive.Create(options)
    (ractive, data)


let stockTickerProcess ractive stocks = 
    let rec waitingLoop (r : Ractive, stocks : List<Stock>) : Async<unit> = 
        async { 
            let ev1, ev2, ev3 = r.onStream ("open", "close", "buyStock")
          
            // Async waiting for an event and then react!!!
            let! choice = Async.AwaitObservable(ev1, ev2, ev3)
          
            match choice with
            | Choice1Of3(ev, arg) -> 
                serverHub.OpenMarket() |> ignore
                j("#open").prop("disabled", true) |> ignore
                j("#close").prop("disabled", false) |> ignore
          
            | Choice2Of3(ev, arg) -> 
                serverHub.CloseMarket() |> ignore
                j("#open").prop("disabled", false) |> ignore
                j("#close").prop("disabled", true) |> ignore
                stopScrollTicker() |> ignore
          
            | Choice3Of3(ev, arg) -> 
                let tickerSymbolBuy = j ("#tickerSymbolBuy")
                let tickerPriceBuy = j ("#tickerPriceBuy")
                let tickerQuantityBuy = j ("#tickerQuantityBuy")
                let symbol : string = tickerSymbolBuy._val() |> unbox
                let priceStr : string = tickerPriceBuy._val() |> unbox
                let quantity : string = tickerQuantityBuy._val() |> unbox
                tickerSymbolBuy._val ("") |> ignore
                tickerPriceBuy._val ("") |> ignore
                tickerQuantityBuy._val ("") |> ignore
               
                let orderRequestBuy = 
                    { ConnectionID = signalR.hub.id
                      Symbol = symbol
                      Quantity = int (quantity)
                      Price = (float priceStr) }
                async { 
                    let url = "http://localhost:48430/api/trading/Buy"
                    let req = System.Net.WebRequest.Create(url)
                    do! req.AsyncPostJSONOneWay(url, orderRequestBuy)
                }
                |> Async.StartImmediate
            return! waitingLoop (r, stocks)
        }
    Async.StartImmediate <| waitingLoop (ractive, stocks)  // Async operation

let orderProcess (ractive : Ractive) = 
    // I can use reactive extanions.. no problem!
    ractive.onStream ("addTicker") |> Observable.add (fun (ev, arg) -> 
                                          let tickerSymbol = j ("#tickerSymbol")
                                          let tickerPrice = j ("#tickerPrice")
                                          let symbol : string = tickerSymbol._val() |> unbox
                                          let priceStr : string = tickerPrice._val() |> unbox
                                          tickerSymbol._val ("") |> ignore
                                          tickerPrice._val ("") |> ignore
                                         
                                          let tikcerRequest = 
                                              { ConnectionID = signalR.hub.id
                                                TikcerRequest.Symbol = symbol
                                                Quantity = 0
                                                Price = (float priceStr) }
                                     
                                          async { 
                                              let url = "http://localhost:48430/api/trading/addTicker"
                                              let req = System.Net.WebRequest.Create(url)
                                              do! req.AsyncPostJSONOneWay(url, tikcerRequest)
                                          }
                                          |> Async.StartImmediate)

    ractive.onStream ("sellStock") |> Observable.add (fun (ev, arg) -> 
                                          let tickerSymbolSell = j ("#tickerSymbolSell")
                                          let tickerPriceSell = j ("#tickerPriceSell")
                                          let tickerQuantitySell = j ("#tickerQuantitySell")
                                          let symbol : string = tickerSymbolSell._val() |> unbox
                                          let priceStr : string = tickerPriceSell._val() |> unbox
                                          let quantity : string = tickerQuantitySell._val() |> unbox
                                          tickerSymbolSell._val ("") |> ignore
                                          tickerPriceSell._val ("") |> ignore
                                          tickerQuantitySell._val ("") |> ignore
                                     
                                          let orderRequestSell = 
                                              { ConnectionID = signalR.hub.id
                                                Symbol = symbol
                                                Quantity = int (quantity)
                                                Price = (float priceStr) }
                                     
                                          async { 
                                              let url = "http://localhost:48430/api/trading/Sell"
                                              let req = System.Net.WebRequest.Create(url)
                                              do! req.AsyncPostJSONOneWay(url, orderRequestSell)
                                          }
                                          |> Async.StartImmediate)

let main() = 
    Globals.console.log("##Starting:## ")
    signalR.hub.url <- "http://localhost:48430/signalr/hubs"

    let client = Hubs.StockTickerHubClient()

    let reactive, data = init()


    // SignalR is sending some notification
    // I am registering the function to handle the notification
    // function are statically typed!!!
    client.SetInitialAsset <- (fun amount -> 
                                    let cash = (sprintf "%.2f" amount)
                                    j("#portfolioCash")._val(cash) |> ignore )

    client.UpdateOrderBuy <- (fun order ->
                                    let buyOrders:List<OrderRecord> = (unbox data.["buyOrders"])
                                    buyOrders.Add(order))
    
    client.UpdateOrderSell <- (fun order ->
                                    let sellOrders:List<OrderRecord> = (unbox data.["sellOrders"])
                                    sellOrders.Add(order))

    client.SetStock <- (fun stock ->                
                 let stocks = ((unbox data.["stocks"]) :> List<Stock>)
                 if not <| (stocks |> Seq.exists(fun t -> t.Symbol.ToUpper() = stock.Symbol.ToUpper())) then 
                    stocks.Add(stock) )
    
    client.UpdateAsset <- (fun asset -> 
                            
                            let buyOrders:List<OrderRecord> = (unbox data.["buyOrders"])
                            let portfolio:List<OrderRecord> = (unbox data.["asset"]) 
                            let sellOrders:List<OrderRecord> = (unbox data.["sellOrders"])

                            let cash = (sprintf "%.2f" asset.Cash)
                            j("#portfolioCash")._val(cash) |> ignore


                        )

    client.UpdateStockPrice <- (fun (stock) -> 
                 ((unbox data.["stocks"]) :> List<Stock>).[stock.Index] <- stock)

    client.Register(signalR.hub)
    
    stockTickerProcess reactive (unbox data.["stocks"])
    orderProcess reactive

    signalR.hub.start onstart

type Wrapper() =
    member this.GenerateScript() = Compiler.compileWithoutReturn <@ main() @>

 

