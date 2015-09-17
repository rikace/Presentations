namespace Easj360FSharp

type Agent<'T> = MailboxProcessor<'T>

//This is the response the supervisor
//gives to the worker request for work
type 'work SupervisorResponse =
| Work of 'work //a piece of work
| NoWork//no work left to do 

//This is the message to the supervisor
type 'work WorkMsg = 
| ToDo of 'work //piles up work in the Supervisor queue
| WorkReq of   AsyncReplyChannel<SupervisorResponse<'work>> //' 

//The supervisor agent can be interacted with
type AgentOperation = 
| Stop //stop the agent
| Status //yield the current status of supervisor

type 'work SupervisorMsg = 
| WorkRel of 'work WorkMsg
| Operation of AgentOperation 

//Supervises Map and Reduce workers
module AgentSupervisor= 
    let getNew (name:string) = 
        new Agent<SupervisorMsg<'work>>(fun inbox -> //'
            let rec loop state  = async {
                let! msg = inbox.Receive()
                match msg with 
                | WorkRel(m) -> 
                    match m with 
                    | ToDo(work) -> 
                        let newState = work:state
                        return! loop newState
                    | WorkReq(replyChannel) ->  
                        match state with 
                        | [] -> 
                            replyChannel.Reply(NoWork)
                            return! loop []
                        | [item] -> 
                            replyChannel.Reply(Work(item))
                            return! loop []
                        | (item::remaining) -> 
                            replyChannel.Reply(Work(item))
                            return! loop remaining
                | Operation(op) -> 
                    match op with 
                    | Status -> 
                        Console.WriteLine(name+" current Work Queue "+
                                            string (state.Length))
                        return! loop state
                    | Stop -> 
                        Console.WriteLine("Stoppped SuperVisor Agent "+name)
                        return()
            }
            loop [] )
    let stop (agent:Agent<SupervisorMsg<'work>>) = agent.Post(Operation(Stop))
    let status (agent:Agent<SupervisorMsg<'work>>) =agent.Post(Operation(Status))

//Code for the workers
type 'success WorkOutcome = 
| Success of 'success
| Fail

type WorkerMsg = 
| Start
| Stop
| Continue

module AgentWorker = 
    type WorkerSupervisors<'reduce,'work> = 
        { Map:Agent<SupervisorMsg<'work>> ; Reduce:Agent<SupervisorMsg<'reduce>> }

    let stop (agent:Agent<WorkerMsg>) = agent.Post(Stop)
    let start (agent:Agent<WorkerMsg>) = agent.Start()
                                         agent.Post(Start)

//    let getNewMapWorker( map, supervisors:WorkerSupervisors<'reduce,'work> ) = 
//        new Agent<WorkerMsg>(fun inbox ->  
//            let rec loop ()  = async {
//                let! msg = inbox.Receive()
//                match msg with 
//                | Start -> inbox.Post(Continue)
//                           return! loop ()
//                | Stop -> 
//                    Console.WriteLine("Map worker stopped")
//                    return ()
//                | Continue ->   
//                    let! supervisorOrder = 
//                    supervisors.Map.PostAndAsyncReply(
//                        fun replyChannel -> 
//                            WorkRel(WorkReq(replyChannel)))
//                    match supervisorOrder with 
//                    | Work(work) -> 
//                        let! res = map work 
//                        match res with
//                        | Success(toReduce) -> 
//                            supervisors.Reduce
//                                .Post(WorkRel(ToDo(toReduce)))
//                        | Fail -> 
//                            Console.WriteLine("Map Fail")
//                            supervisors.Map
//                                .Post(WorkRel(ToDo(work)))
//                            inbox.Post(Continue)
//                    | NoWork -> 
//                            inbox.Post(Continue)
//                            return! loop ()               
//                  }
//            loop ())


    let getNewReduceWorker(reduce,reduceSupervisor:Agent<SupervisorMsg<'work>>)= 
        new Agent<WorkerMsg>(fun inbox ->  
            let rec loop ()  = async {
                let! msg = inbox.Receive()
                match msg with
                | Start -> inbox.Post(Continue)
                           return! loop()
                | Continue ->   
                    let! supervisorOrder = 
                        reduceSupervisor.PostAndAsyncReply(fun replyChannel -> 
                            WorkRel(WorkReq(replyChannel)))
                    match supervisorOrder with 
                    | Work(work) -> 
                        let! res = reduce work 
                        match res with 
                        | Success(toReduce) -> inbox.Post(Continue)
                        | Fail -> 
                            Console.WriteLine("ReduceFail")
                            reduceSupervisor.Post(WorkRel(ToDo(work)))
                            inbox.Post(Continue)
                    | NoWork -> inbox.Post(Continue)
                    return! loop()
                |Stop ->Console.WriteLine("Reduce worker stopped"); return () 
                }
            loop() )

open AgentWorker

type MapReduce<'work,'reduce>( numberMap:int , 
                               numberReduce: int, 
                               toProcess:'work list,  
                               map:'work->Async<'reduce WorkOutcome>,
                               reduce:'reduce-> Async<unit WorkOutcome>) = 

    let mapSupervisor= AgentSupervisor.getNew("MapSupervisor")  
    let reduceSupervisor  = AgentSupervisor.getNew("ReduceSupervisor")

    let workerSupervisors = {Map = mapSupervisor ; Reduce = reduceSupervisor }

    let mapWorkers = 
        [for i in 1..numberMap -> 
            AgentWorker.getNewMapWorker(map,workerSupervisors) ]
    let reduceWorkers = 
        [for i in 1..numberReduce -> 
            AgentWorker.getNewReduceWorker(reduce,workerSupervisors.Reduce) ] 

    member this.Start() = 
        //Post work to do
        toProcess
        |>List.iter(fun elem -> mapSupervisor.Post( WorkRel(ToDo(elem))))
        //Start supervisors
        mapSupervisor.Start()
        reduceSupervisor.Start()
        //start workers 
        List.iter( fun mapper -> mapper |>start) mapWorkers 
        List.iter( fun reducer ->reducer|>start) reduceWorkers

    member this.Status() =  (mapSupervisor|>AgentSupervisor.status)
                            (reduceSupervisor|>AgentSupervisor.status)
    member this.Stop() = 
        List.map2(fun mapper reducer -> 
            mapper |>stop; reducer|>stop) mapWorkers reduceWorkers

//Run some tests
let map = function (n:int64) -> async{ return Success(n) } 

let reduce = function (toto: int64) -> async{ return Success() }

let mp = MapReduce<int64,int64>( 1,1,[for i in 1L..1000000L->i],map,reduce)

mp.Start()
mp.Status()
mp.Stop()


#r "System.Xml.Linq.dll"

#r "FSharp.PowerPack.dll"

open System
open System.Diagnostics
open System.IO
open System.Linq
open System.Net
open System.Xml.Linq

open Microsoft.FSharp.Control.WebExtensions 

type Weather (city, region, temperature) = class
   member x.City = city
   member x.Region = region
   member x.Temperature : int = temperature

   override this.ToString() =
      sprintf "%s, %s: %d F" this.City this.Region this.Temperature
end

type MessageForActor = 
   | ProcessWeather of Weather
   | ProcessError of int
   | GetResults of (Weather * Weather * Weather list) AsyncReplyChannel

let parseRss woeid (rssStream : Stream) =
   let xn str = XName.Get str
   let yweather elementName = XName.Get(elementName, "http://xml.weather.yahoo.com/ns/rss/1.0")

   let channel = (XDocument.Load rssStream).Descendants(xn "channel").First()
   let location   = channel.Element(yweather "location")
   let condition  = channel.Element(xn "item").Element(yweather "condition")

   //  If the RSS server returns error, condition XML element won't be available.
   if not(condition = null) then
      let temperature = Int32.Parse(condition.Attribute(xn "temp").Value)
      ProcessWeather(new Weather(
                    location.Attribute(xn "city").Value,
                    location.Attribute(xn "region").Value,
                    temperature))
   else
      ProcessError(woeid)

let fetchWeather (actor : MessageForActor MailboxProcessor) woeid =
   async {
      let rssAddress = sprintf "http://weather.yahooapis.com/forecastrss?w=%d&u=f" woeid
      let webRequest =  WebRequest.Create rssAddress
      use! response = webRequest.AsyncGetResponse()
      use responseStream = response.GetResponseStream()
      let weather = parseRss woeid responseStream
      //do! Async.Sleep 1000 // enable this line to see amplified timing that proves concurrent flow
      actor.Post(weather)
   }

let mailboxLoop initialCount =
   let chooseCityByTemperature op (x : Weather) (y : Weather) =
      if op x.Temperature y.Temperature then x else y

   let sortWeatherByCityAndState (weatherList : Weather list) =
      weatherList
      |> List.sortWith (fun x y -> x.City.CompareTo(y.City))
      |> List.sortWith (fun x y -> x.Region.CompareTo(y.Region))

   MailboxProcessor.Start(fun inbox ->
      let rec loop minAcc maxAcc weatherList remaining =
         async {
            let! message = inbox.Receive()
            let remaining = remaining - 1

            match message with
            | ProcessWeather weather ->
               let colderCity = chooseCityByTemperature (<) minAcc weather
               let warmerCity = chooseCityByTemperature (>) maxAcc weather
               return! loop colderCity warmerCity (weather :: weatherList) remaining
            | ProcessError woeid ->
               let errorWeather = new Weather(sprintf "Error with woeid=%d" woeid, "ZZ", 99999)
               return! loop minAcc maxAcc (errorWeather :: weatherList) remaining
            | GetResults replyChannel ->
               replyChannel.Reply(minAcc, maxAcc, sortWeatherByCityAndState weatherList)
         }

      let minValueInitial = new Weather("", "", Int32.MaxValue)
      let maxValueInitial = new Weather("", "", Int32.MinValue)
      loop minValueInitial maxValueInitial [] initialCount
      )

let RunSynchronouslyWithExceptionAndTimeoutHandlers computation =
   let timeout = 30000
   try
      Async.RunSynchronously(Async.Catch(computation), timeout)
      |> function Choice1Of2 answer               -> answer |> ignore
                | Choice2Of2 (except : Exception) -> printfn "%s" except.Message; printfn "%s" except.StackTrace; exit -4
   with
   | :? System.TimeoutException -> printfn "Timed out waiting for results for %d seconds!" <| timeout / 1000; exit -5

let main =
   // Should have script name, sync/async select, and at least one woeid
   if fsi.CommandLineArgs.Length < 3 then
      printfn "Expecting at least two arguments!"
      printfn "There were %d arguments" (fsi.CommandLineArgs.Length - 1)
      exit -1

   let woeids =
      try
         fsi.CommandLineArgs
         |> Seq.skip 2 // skip the script name and sync/async select
         |> Seq.map Int32.Parse
         |> Seq.toList
      with
      | except -> printfn "One of supplied arguments was not an integer: %s" except.Message; exit -2

   let actor = mailboxLoop woeids.Length

   let processWeatherItemsConcurrently woeids =
      woeids
      |> Seq.map (fetchWeather actor)
      |> Async.Parallel
      |> RunSynchronouslyWithExceptionAndTimeoutHandlers

   let processOneWeatherItem woeid =
      woeid
      |> fetchWeather actor
      |> RunSynchronouslyWithExceptionAndTimeoutHandlers

   let stopWatch = new Stopwatch()
   stopWatch.Start()
   match fsi.CommandLineArgs.[1].ToUpper() with
   | "C" -> printfn "Concurrent execution:  "; processWeatherItemsConcurrently woeids
   | "S" -> printfn "Synchronous execution: "; woeids |> Seq.iter processOneWeatherItem
   | _   -> printfn "Unexpected run options!"; exit -3

   let (min, max, weatherList) = actor.PostAndReply GetResults
   stopWatch.Stop()
   assert (weatherList.Length = woeids.Length)

   printfn "{"
   weatherList |> List.iter (printfn "   %O")
   printfn "}"
   printfn "Coldest place: %O" min
   printfn "Hottest place: %O" max
   printfn "Completed in %d millisec" stopWatch.ElapsedMilliseconds

main
