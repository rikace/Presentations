namespace StockTicker


[<AutoOpenAttribute>]
module Rop =

    type Error =
        | InvalidState of string
        | InvalidStateTransition of string
        | NotSupportedCommand of string
        | UnknownDto of string
        | ValidationError of string
        | InvalidPaymentAmount


    // the two-track type
    type Result<'TSuccess,'TFailure> =
        | Success of 'TSuccess
        | Failure of 'TFailure


    // apply either a success function or failure function
    let inline either successFunc failureFunc twoTrackInput =
        match twoTrackInput with
        | Success s -> successFunc s
        | Failure f -> failureFunc f

    // convert a switch function into a two-track function
    let inline bind f twoTrackInput =
        either f Failure twoTrackInput




