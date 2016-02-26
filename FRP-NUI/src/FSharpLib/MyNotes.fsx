namespace Easj360

[<AutoOpenAttribute>]
module MyNotes =

    open System
    open System.IO
    open System.Threading
    
    type RequestGate(n : int) =
        let semaphore = new Semaphore(initialCount = n, maximumCount = n)
        member x.AsyncAcquire(?timeout) =
            async {let! ok = Async.AwaitWaitHandle(semaphore,
                                                   ?millisecondsTimeout = timeout)
                   if ok then
                       return
                         {new System.IDisposable with
                             member x.Dispose() =
                                 semaphore.Release() |> ignore}
                   else
                       return! failwith "couldn't acquire a semaphore" }

    type System.Timers.Timer with 
        static member StartWithDisposable interval handler = 
            // create the timer
            let timer = new System.Timers.Timer(interval)
            
            // add the handler and start it
            do timer.Elapsed.Add handler 
            timer.Start()
            
            // return an IDisposable that calls "Stop"
            { new System.IDisposable with 
                member disp.Dispose() = 
                    do timer.Stop() 
                    do printfn "Timer stopped"
                }


    let using (ie : #System.IDisposable) f =
        try f(ie)
        finally ie.Dispose()
    //val using : ie:'a -> f:('a -> 'b) -> 'b when 'a :> System.IDisposable

    let rec last l =
        match l with
        | [] -> invalidArg "l" "the input list should not be empty"
        | [h] -> h
        | h::t -> last t

module MyComputationExpression =
    open System

    type RoundComputationBuilder(digits:int) =
        let round (x:decimal) = System.Math.Round(x,digits)
        member this.Bind(result, restComputation) =
            restComputation(round result)
        member this.Return x = round x

    let round = RoundComputationBuilder 2
            
    round {
        let! x = 2323.34m * 0.0002m
        return x }



    type Retry (retryTimes:int) = class
        let mutable success = false
        member public this.Bind(value, restFunction:unit -> _) =
            success <- false
            let mutable n = retryTimes
            while not success && n > 0 do
                n <- n - 1
                try
                    value()
                    success <- true
                with
                | _ as e -> printfn "Error %s" e.Message
            restFunction()
        member this.Return args = success
    end

    let retry = Retry 2

    let a() = retry {
        do! (fun () ->  printfn "Retry"
                        failwith "test Failure")
        }

    a()
module Alghos =
    
    let rec quicksort list =
       match list with
       | [] ->                            // If the list is empty
            []                            // return an empty list
       | firstElem::otherElements ->      // If the list is not empty     
            let smallerElements =         // extract the smaller ones    
                otherElements             
                |> List.filter (fun e -> e < firstElem) 
                |> quicksort              // and sort them
            let largerElements =          // extract the large ones
                otherElements 
                |> List.filter (fun e -> e >= firstElem)
                |> quicksort              // and sort them
            // Combine the 3 parts into a new list and return it
            List.concat [smallerElements; [firstElem]; largerElements]

        //test ->        printfn "%A" (quicksort [1;5;23;18;9;1;3])

    let rec quicksort2 = function
       | [] -> []                         
       | first::rest -> 
            let smaller,larger = List.partition ((>=) first) rest 
            List.concat [quicksort2 smaller; [first]; quicksort2 larger]
        
    // test code        -> printfn "%A" (quicksort2 [1;5;23;18;9;1;3])


    let rec interpolation ls =
        match ls with
        | [] -> []
        | x::y::rest ->
            let avg = (x+y)/2.0
            x :: avg :: y :: (interpolation rest)
        | x::y -> x::((x+y)/2.0)::y

module StateMachineModule = 
    type States = 
    | State1
    | State2
    | State3

    type StateMachine() = 
        let stateMachine = new MailboxProcessor<States>(fun inbox ->
                    let rec state1 () = async {
                        printfn "current state is State1"
                        // <your operations>

                        //get another message and perform state transition
                        let! msg = inbox.Receive()
                        match msg with
                            | State1 -> return! (state1())
                            | State2 -> return! (state2())
                            | State3 -> return! (state3())
                        }
                    and state2() = async {
                        printfn "current state is state2"
                        // <your operations>

                        //get another message and perform state transition
                        let! msg = inbox.Receive()
                        match msg with
                            | State1 -> return! (state1())
                            | State2 -> return! (state2())
                            | State3 -> return! (state3())
                        }
                    and state3() = async {
                        printfn "current state is state3"
                        // <your operations>

                        //get another message and perform state transition
                        let! msg = inbox.Receive()
                        match msg with
                            | State1 -> return! (state1())
                            | State2 -> return! (state2())
                            | State3 -> return! (state3())
                        } 
                    and state0 () = 
                        async {

                            //get initial message and perform state transition
                            let! msg = inbox.Receive()
                            match msg with
                                | State1 -> return! (state1())
                                | State2 -> return! (state2())
                                | State3 -> return! (state3())
                        }
                    state0 ())

        //start the state machine and set it to state0
        do 
            stateMachine.Start()        

        member this.ChangeState(state) = stateMachine.Post(state)

    let stateMachine = StateMachine()
    stateMachine.ChangeState(States.State2)
    stateMachine.ChangeState(States.State1)


module AsyncHelpers = 

    let RunSynchronouslyWithExceptionAndTimeoutHandlers computation =
       let timeout = 30000
       try
          Async.RunSynchronously(Async.Catch(computation), timeout)
          |> function Choice1Of2 answer               -> answer |> ignore
                    | Choice2Of2 (except : Exception) -> printfn "%s" except.Message; printfn "%s" except.StackTrace; exit -4
       with
       | :? System.TimeoutException -> printfn "Timed out waiting for results for %d seconds!" <|

    let userTimerWithAsync = 
        // create a timer and associated async event
        let timer = new System.Timers.Timer(2000.0)
        let timerEvent = Async.AwaitEvent (timer.Elapsed) |> Async.Ignore
        // start
        printfn "Waiting for timer at %O" DateTime.Now.TimeOfDay
        timer.Start()
        // keep working
        printfn "Doing something useful while waiting for event"
        // block on the timer event now by waiting for the async to complete
        Async.RunSynchronously timerEvent
        // done
        printfn "Timer ticked at %O" DateTime.Now.TimeOfDay


    let fileWriteWithAsync = 
        // create a stream to write to
        use stream = new System.IO.FileStream("test.txt",System.IO.FileMode.Create)
        // start
        printfn "Starting async write"
        let asyncResult = stream.BeginWrite(Array.empty,0,0,null,null)
        // create an async wrapper around an IAsyncResult
        let async = Async.AwaitIAsyncResult(asyncResult) |> Async.Ignore
        // keep working
        printfn "Doing something useful while waiting for write to complete"
        // block on the timer now by waiting for the async to complete
        Async.RunSynchronously async 
        // done
        printfn "Async write completed"


module WebCrawler = 
    open System.Collections.Generic
    open System.Net
    open System.IO
    open System.Threading
    open System.Text.RegularExpressions

    let limit = 50
    let linkPat = "href=\s*\"[^\"h]*(http://[^&\"]*)\""
    let getLinks (txt : string) =
        [for m in Regex.Matches(txt, linkPat) -> m.Groups.Item(1).Value]

    // A type that helps limit the number of active web requests
    type RequestGate(n : int) =
        let semaphore = new Semaphore(initialCount = n, maximumCount = n)
        member x.AsyncAcquire(?timeout) =
            async {let! ok = Async.AwaitWaitHandle(semaphore,
                                                   ?millisecondsTimeout = timeout)
                   if ok then
                       return
                         {new System.IDisposable with
                             member x.Dispose() =
                                 semaphore.Release() |> ignore}
                   else
                       return! failwith "couldn't acquire a semaphore" }

    // Gate the number of active web requests
    let webRequestGate = RequestGate(5)

    // Fetch the URL, and post the results to the urlCollector.
    let collectLinks (url : string) =
        async { // An Async web request with a global gate
                let! html = async {
                    // Acquire an entry in the webRequestGate. Release
                    // it when 'holder' goes out of scope
                    use! holder = webRequestGate.AsyncAcquire()

                    let req = WebRequest.Create(url, Timeout = 5)

                    // Wait for the WebResponse
                    use! response = req.AsyncGetResponse()

                    // Get the response stream
                    use reader = new StreamReader(response.GetResponseStream())

                    // Read the response stream (note: a synchronous read)
                    return reader.ReadToEnd()}

                // Compute the links, synchronously
                let links = getLinks html

                // Report, synchronously
                do printfn "finished reading %s, got %d links" url (List.length links)

                // We're done
                return links}

    /// 'urlCollector' is a single agent that receives URLs as messages. It creates new
    /// asynchronous tasks that post messages back to this object.
    let urlCollector =
        MailboxProcessor.Start(fun self ->

            // This is the main state of the urlCollector
            let rec waitForUrl (visited : Set<string>) = async {
                // Check the limit
                if visited.Count < limit then

                    // Wait for a URL...
                    let! url = self.Receive()
                    if not (visited.Contains(url)) then
                        // Start off a new task for the new url. Each collects
                        // links and posts them back to the urlCollector.
                        do! Async.StartChild (async {
                            let! links = collectLinks url
                            for link in links do
                                self.Post link}) |> Async.Ignore

                    // Recurse into the waiting state
                    return! waitForUrl(visited.Add(url))}

            // This is the initial state.
            waitForUrl(Set.empty))
    //val limit : int = 50
    //val linkPat : string = "href=\s*"[^"h]*(http://[^&"]*)""
    //val getLinks : txt:string -> string list
    //type RequestGate =
    //  class
    //    new : n:int -> RequestGate
    //    member AsyncAcquire : ?timeout:int -> Async<System.IDisposable>
    //  end
    //val webRequestGate : RequestGate
    //val collectLinks : url:string -> Async<string list>
    //val urlCollector : MailboxProcessor<string>

    > urlCollector <-- "http://news.google.com";;
    > urlCollector.Post "http://news.google.com";;
    //finished reading http://news.google.com, got 191 links
    //finished reading http://news.google.com/?output=rss, got 0 links
    //finished reading http://www.ktvu.com/politics/13732578/detail.html, got 14 links
    //finished reading http://www.washingtonpost.com/wp-dyn/content/art..., got 218 links
    //finished reading http://www.newsobserver.com/politics/story/646..., got 56 links
    //finished reading http://www.foxnews.com/story/0,2933,290307,0...l, got 22 links
    //...

module TestMyNotes =
    let testUsing = using