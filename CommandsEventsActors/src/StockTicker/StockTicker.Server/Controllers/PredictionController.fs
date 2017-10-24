namespace StockTicker.Controllers

open System
open System.Net
open System.Net.Http
open System.Web.Http
open StockTicker
open StockTicker.Core
open StockTicker.Logging
open StockTicker.Logging.Message

[<RoutePrefix("api/prediction")>]
type PredictionController() =
    inherit ApiController()

    static let logger = Log.create "StockTicker.PredictionController"

    [<Route("predict"); HttpPost>]
    member this.PostPredict([<FromBody>] pr : PredictionRequest) =
        async {
            // Log prediction request
            do! logger.logWithAck Info (
                 eventX "{logger}: Called {url} with request {pr}"
                 >> setField "logger" (sprintf "%A" logger.name)
                 >> setField "url" ("/api/prediction/predict")
                 >> setField "pr" ((sprintf "%A" pr).Replace("\n","")) )

            // Run simulation / do prediction
            let volatility = Simulations.Volatilities
                             |> Map.tryFind pr.Symbol
            let prediction =
                { MeanPrice =
                    match volatility with
                    | None -> pr.Price
                    | Some(x) ->
                        Simulations.calcPriceCPU <|
                            { TimeSteps = pr.NumTimesteps
                              Price = pr.Price
                              Volatility = x}
                  Quartiles = [||]}

            // Log simulation result
            do! logger.logWithAck Info (
                 eventX "{logger}: Prediction for {sym} => {resp}"
                 >> setField "logger" (sprintf "%A" logger.name)
                 >> setField "sym"  (pr.Symbol)
                 >> setField "resp" ((sprintf "%A" prediction).Replace("\n","")) )

            // Return response
            return this.Request.CreateResponse(HttpStatusCode.OK, prediction);
        } |> Async.StartAsTask
