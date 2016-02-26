namespace FsFRPLib

open Core    
open System
open System.Threading
open System.Threading.Tasks


[<RequireQualifiedAccessAttribute>]
module Reactivity =

    let unamb (taskOne : (unit -> 'a)) (taskTwo:(unit ->'a)) = async {
            let! cnlToken = Async.CancellationToken             
            
            let parentCancelToken = CancellationTokenSource.CreateLinkedTokenSource cnlToken

            return! Async.FromContinuations(fun(ccont, cexn, ccnl) ->

                 let execute task = async {return task () }

                 ignore <| Task.Run(fun () -> 
                        Async.StartWithContinuations(execute taskOne, 
                            (fun result -> parentCancelToken.Cancel()
                                           ccont result),
                            (fun (error:exn) ->
                                parentCancelToken.Cancel()
                                cexn error),
                            (fun (cancel : OperationCanceledException) ->
                                ccnl cancel), parentCancelToken.Token))
               
                 ignore <| Task.Run(fun () -> 
                        Async.StartWithContinuations(execute taskTwo, 
                            (fun result -> parentCancelToken.Cancel()
                                           ccont result),
                            (fun (error:exn) ->
                                parentCancelToken.Cancel()
                                cexn error),
                            (fun (cancel : OperationCanceledException) ->
                                ccnl cancel), parentCancelToken.Token))) }

  
  
    let unambBehavior (taskOne : (unit -> 'a)) (taskTwo:(unit ->'a)) =
        let tcs = new TaskCompletionSource<Behavior<'a>>()
        ignore <| Task.Run(fun () ->
            try
                 tcs.SetResult (pureBehavior (unamb taskOne taskTwo |> Async.RunSynchronously))
            with
            | ex -> tcs.SetException ex ) 
        let rec someBeh v = Behavior(fun t -> (v, fun () -> someBeh v))
        ignore <| tcs.Task.ContinueWith(fun (t:Task<Behavior<'a>>) -> 
                                                let beh = t.Result
                                                someBeh (Some beh))
        someBeh None

    let unambEvent (taskOne : (unit -> 'a)) (taskTwo:(unit ->'a)) =
        pureEvent (unamb taskOne taskTwo |> Async.RunSynchronously)



module testUnam =

    
          
    let delay interval result =
        async {
            do! Async.Sleep interval
            return! async {
                printfn "returning %A after %d ms." result interval
                return result }
        } |> Async.RunSynchronously

    Reactivity.unamb (fun () -> delay 100 "test A") (fun () -> delay 1000 "Test B") 
        |> Async.RunSynchronously  |> fun r -> printfn "result %A" r        
  //  (fun () -> delay 100 "test A"), (fun () -> delay 1000 "Test B") ||> Async.unamb 
    


