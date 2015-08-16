//                                
//     /\                         
//    /  \   ___ _   _ _ __   ___ 
//   / /\ \ / __| | | | '_ \ / __|
//  / ____ \\__ \ |_| | | | | (__ 
// /_/    \_\___/\__, |_| |_|\___|
//                __/ |           
//               |___/            

(*  Asynchronous programming is when you begin an operation on a background 
    thread to execute concurrently, and have it terminate at some later time; 
    either ignoring the new task to complete in the background, or polling it 
    at a later time to check on its return value    *)
open System
open System.IO
open System.Threading
open System.Net

// asycns are first-class, parallelism by combining asyncs
module AsyncBasic =  

    let asyncResult = 
        async {
            let wc = new WebClient()
            let! content = wc.AsyncDownloadString(Uri("http://www.comp3020.com"))
            return content.Length
        }


    let AsyncAdd5 original =
        async {
            let! a = original
            return a + 5
        } |> Async.RunSynchronously


    

    let contentLength = AsyncAdd5 asyncResult


    let prog(i) = async {   printfn "Starting %d ..." i
                            do! Async.Sleep((i%5) * 1000)
                            printfn "Finishing %d ..." i
                            return 1 }

    let result = 
        [1..10]
        //[1..100000]  // asyncs are very cheap, 100000 running concurrently, only a couple of threads used
        |> Seq.map (fun i -> prog(i))
        |> Async.Parallel
        |> Async.StartAsTask


    let sleep name seconds = async {
        System.Threading.Thread.Sleep(seconds * 1000)
        printfn "%s slept %d seconds" name seconds }

    let task1 = sleep "Task 1" 4
    let task2 = sleep "Task 2" 3
    let task3 = sleep "Task 3" 2

    [ task1; task2; task3 ] // declare a list on the fly and push into Async.Parallel
      |> Async.Parallel
      |> Async.RunSynchronously

    let urls = [ "http://www.meetup.com/DC-fsharp"; 
                 "http://www.fsharp.org"; 
                 "http://www.google.com"]
    
    // make it Async !
    urls 
    |> List.map (fun url -> 
            let client = new System.Net.WebClient()
            let html = client.DownloadString(new System.Uri(url))
            printfn "Site len %d" html.Length )    
    
    //|> Async.Parallel
    

    // Fetch the contents of a web page asynchronously
    let fetchUrlAsync url =        
        async {                             
            let req = WebRequest.Create(Uri(url)) 
            use! resp = req.AsyncGetResponse()  // new keyword "use!"  
            use stream = resp.GetResponseStream() 
            use reader = new IO.StreamReader(stream) 
            let html = reader.ReadToEnd() 
            printfn "finished downloading %s" url 
            }
        // a list of sites to fetch
    let sites = ["http://www.bing.com";
                 "http://www.google.com";
                 "http://www.microsoft.com";
                 "http://www.amazon.com";
                 "http://www.yahoo.com"]

    sites 
    |> List.map fetchUrlAsync  // make a list of async tasks
    |> Async.Parallel          // set up the tasks to run in parallel
    |> Async.RunSynchronously  // start them off

    
////////////// Async StartWithContinuation

let downloadComp (url : string) = async {
        let req = WebRequest.Create(url)
        let! rsp = req.AsyncGetResponse()
        use stream = rsp.GetResponseStream()
        use reader = new StreamReader(stream)
        return! reader.ReadToEndAsync() |> Async.AwaitTask }


let okCon (s: string) = printf "Length = %d\n" (s.Length) 
let exnCon _ = printf "Exception raised\n" 
let canCon _ = printf "Operation cancelled\n"

let cancelExample() =
        use ts = new CancellationTokenSource()
     
        Async.StartWithContinuations
            (downloadComp "http://www.microsoft.com",
             okCon, exnCon, canCon, ts.Token)
      
        ts.Cancel()  

cancelExample()


(*  In asynchronous workflows, unhandled exceptions bring down the whole process 
    because by default they are not caught. To catch unhandled exceptions from 
    async workflows, you can use the Async.StartWithContinuations method, 
    discussed later, or use the Async.Catch combinator.     *)

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


(*  When you execute code asynchronously, it is important to have a 
    cancellation mechanism just in case... *)

(*  If you want to be able to cancel an arbitrary asynchronous workflow, 
    then you’ll want to create and keep track of a CancellationTokenSource object. 
    A CancellationTokenSource is what signals the cancellation, 
    which in turn updates all of its associated CancellationTokens  *)


open System
open System.Threading

let ct = new System.Threading.CancellationTokenSource()
let asyncComputation = downloadComp "http://www.microsoft.com"

Async.Start(asyncTask, ct.Token)

ct.Cancel()


type Microsoft.FSharp.Control.Async with
  /// Starts the specified operation using a new CancellationToken and returns
  /// IDisposable object that cancels the computation. This method can be used
  /// when implementing the Subscribe method of IObservable interface.

  static member StartDisposable(op:Async<unit>, (?cancelHandler:OperationCanceledException -> unit)) =   
    let ct = new System.Threading.CancellationTokenSource()
    match cancelHandler with
    | None -> Async.Start(op, ct.Token)
    | Some(c) -> let computation = Async.TryCancelled(op, c)
                 Async.Start(computation, ct.Token)
    { new IDisposable with 
        member x.Dispose() = ct.Cancel() }



let cancelableTask =    async { printfn "Waiting 10 seconds..."
                                for i = 1 to 10 do
                                    printfn "%d..." i
                                    do! Async.Sleep(1000)
                                printfn "Finished!" }

// Callback used when the operation is canceled
let cancelHandler (ex : OperationCanceledException) =
    printfn "The task has been canceled."

let asyncDisposable = Async.StartDisposable(cancelableTask, cancelHandler)

asyncDisposable.Dispose()

