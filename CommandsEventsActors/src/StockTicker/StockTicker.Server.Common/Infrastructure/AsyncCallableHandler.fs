namespace WebApp


module Handlers =

    open System.Threading.Tasks
    open System
    open System.Net.Http

    type internal AsyncHandler (handler) =
        inherit DelegatingHandler(handler)
        member internal x.CallSendAsync(request, cancellationToken) =
            base.SendAsync(request, cancellationToken)