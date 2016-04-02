(*  In asynchronous workflows, unhandled exceptions bring down the whole process 
    because by default they are not caught. To catch unhandled exceptions from 
    async workflows, you can use the Async.StartWithContinuations method, 
    discussed later, or use the Async.Catch combinator.     *)
open System
open System.IO

let asncOperation = async {     try
                                    failwith "Error!!"
                                with
                                | :? IOException as ioe ->
                                    printfn "IOException: %s" ioe.Message
                                | :? ArgumentException as ae ->
                                    printfn "ArgumentException: %s" ae.Message  }

let asyncTask = async { raise <| new System.Exception("My Error!") }

asyncTask
|> Async.Catch
|> Async.RunSynchronously
|> function
   | Choice1Of2 result     -> printfn "Async operation completed: %A" result
   | Choice2Of2 (ex : exn) -> printfn "Exception thrown: %s" ex.Message
                      
let asyncTask2 (x:int) = async {  if x % 2 = 0 then
                                      return x 
                                  else return failwith "My Error" }

let run task =  Async.StartWithContinuations(   task, 
                                            (fun result -> printfn "result %A" result),
                                            (fun ex -> printfn "Error %s" ex.Message),
                                            (fun cancel -> printfn "task cancelled"))
run (asyncTask2 6)
run (asyncTask2 5)


let runWithCatch task =
    task
    |> Async.Catch
    |> Async.RunSynchronously
    |> function
        |Choice1Of2 result -> printf "result %A" result
        |Choice2Of2 ex -> printfn "ErrorMessge: %s" ex.Message

runWithCatch (asyncTask2 4)
runWithCatch (asyncTask2 5)