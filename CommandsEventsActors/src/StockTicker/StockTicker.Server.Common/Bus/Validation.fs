namespace StockTicker.Validation

open System
open System
open StockTicker.Rop
open StockTicker.Core
open System.Threading.Tasks

[<AutoOpen>]
module Validation =


    let validateTicker  (input : TradingRecord) =
        if input.Symbol = "" then Failure "Ticket must not be blank"
        else Success input

    let validateQuantity (input : TradingRecord) =
        if input.Quantity <= 0 || input.Quantity > 50 then Failure "Quantity must be positive and not be more than 50"
        else Success input

    let validatePrice (input : TradingRecord) =
        if input.Price <= 0. then Failure "Price must be positive"
        else Success input

    let tradingdValidation =
        validateTicker >> bind validatePrice >> bind validateQuantity





    let validateTickerRequestSymbol (input : TickerRecord) =
        if input.Symbol = "" then Failure "Ticket must not be blank"
        else Success input

    let validateTickerRequestPriceMin (input : TickerRecord) =
        if input.Price <= 0. then Failure "Price must be positive"
        else Success input

    let validateTickerRequestPriceMax (input : TickerRecord) =
        if input.Price >= 1000. then Failure "Price must be positive"
        else Success input

    let tickerRequestValidation =
        validateTickerRequestSymbol >> bind validateTickerRequestPriceMin >> bind validateTickerRequestPriceMax



////////////////////////////


    let validateTickerWithoutRop input =
        match validateTicker input with
        | Failure(e) -> Failure e
        | Success(s) -> match validateQuantity input with
                        | Failure(e) -> Failure e
                        | Success(s) -> match validatePrice input with
                                        | Failure(e) -> Failure e
                                        | Success(s) -> Success s

    // The bind function here calls the function f1 with the given input x
    // if it succeeds then executes the second function f2 with x.
    // In case if the execution of f1 resulted in failure then
    // it just passes without executing the second function.

    // (f1: 'a -> Result<'b, 'c>)  (f2: 'b -> Result<'d, 'c>)
    let bind f1 f2 x =
      match f1 x with
      | Success x -> f2 x
      | Failure err -> Failure err

    // The infix operator >>= is an alias of bind function.
    let inline (>>=) f1 f2 = bind f1 f2





