#r "../../packages/Rx-Core.2.2.5/lib/net45/System.Reactive.Core.dll"
#r "../../packages/Rx-Interfaces.2.2.5/lib/net45/System.Reactive.Interfaces.dll"
#r "../../packages/Rx-Linq.2.2.5/lib/net45/System.Reactive.Linq.dll"
#r "../../packages/Rx-Xaml.2.2.5/lib/net45/System.Reactive.Windows.Threading.dll"
#load "Utils.fs"

open System
open System.Reactive
open System.Reactive.Linq
open System.Reactive.Subjects
open System.Threading
open System.Threading.Tasks
open System.Collections.Generic
open AgentModule

// ===========================================
// RX Stocks with Subject
// ===========================================
 
let msft = Stock.CreateStock("MSFT") 95.
let amzn = Stock.CreateStock("AMZ") 197.
let goog = Stock.CreateStock("GOOG") 513.

let seqStocks = [msft;amzn;goog]  



let sb = new Subject<Stock>()


let updatedStocks (stocks:Stock list) =
        stocks |> List.map(fun s -> s.Update())

let obs = { new IObserver<Stock> with
                member x.OnNext(s) = printfn "Stock %s - price %4f" s.Symbol s.Price
                member x.OnCompleted() = printfn "Completed"
                member x.OnError(exn) = ()   }

let dispose = sb.Subscribe(obs)   
    
    
let stocksObservable =
    Observable.Interval(TimeSpan.FromMilliseconds (getThreadSafeRandom() * 100.))
    |> Observable.scan(fun s i -> updatedStocks s) seqStocks
    |> Observable.subscribe(fun s -> s |> List.iter (sb.OnNext))

sb.OnCompleted()

stocksObservable.Dispose()
    