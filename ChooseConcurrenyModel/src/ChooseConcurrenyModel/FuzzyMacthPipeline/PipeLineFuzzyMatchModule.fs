module PipeLineFuzzyMatchModule



open System
open System.IO
open System.Net
open System.Threading.Tasks
open ParallelizingFuzzyMatch
open FuzzyMatch
open DataFlowPipeLine
open System.Threading
open System.Threading.Tasks.Dataflow

let urls = Data.GetTextFileUrls()
let cts = new CancellationTokenSource()


let finalAgent = MailboxProcessor<string * string[]>.Start((fun inbox ->
            
            let rec loop state = async {

                let! (name,matches) = inbox.Receive()

                match name, matches with
                | n, [||] -> return! loop state
                | n, m -> 
                
                    let backupColor = Console.ForegroundColor
                    Console.ForegroundColor <- ConsoleColor.Blue
                    printfn "Received Back result for Text %s" name
                    Console.ForegroundColor <- backupColor

                    return! loop (state + 1)
            }
            loop 0), cts.Token)

let consoleAgent = 
        let agent = 
            new MailboxProcessor<int * string>(fun inbox -> 
                let rec loop() = async {
                        let! (step, message) = inbox.Receive()
                        let backupColor = Console.ForegroundColor
                        match step with
                        | 1 ->  Console.ForegroundColor <- ConsoleColor.Red
                        | 2 ->  Console.ForegroundColor <- ConsoleColor.Yellow
                        | 3 ->  Console.ForegroundColor <- ConsoleColor.Green
                        | 4 ->  Console.ForegroundColor <- ConsoleColor.Magenta
                        | _ ->  Console.ForegroundColor <- ConsoleColor.Cyan
                        printfn "%s" message 
                        Console.ForegroundColor <- backupColor
                        return! loop() }
                loop())
        agent.Error.Add(fun err -> printfn "ERROR : %s" err.Message)
        agent.Start()
        agent           
           


let fmdf = FuzzyMatchDataFlow(finalAgent, consoleAgent,cts)


async { do! fmdf.ProcessAsynchronously() 
            |> Async.AwaitTask } 
            |> Async.Start


let downloadTextAsync (url:string) : Async<string> = async {
    use client = new WebClient()
    let headerText = "Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.1; Trident/6.0)"
    client.Headers.Add("user-agent", headerText)
    return! client.AsyncDownloadString(Uri(url)) }


// file processes as arrived Async
// hard to achieve in C# async due to 
// lack of compositionality
let loadDataAsyncInParalallAndProcess() =
    let asyncComp : Async<unit []> =
        urls
        |> Seq.map(fun kv -> async {    let! text = downloadTextAsync kv.Value
                                        printfn "Sending Text %s" kv.Key

                                        let! send = fmdf.InputBlock.SendAsync((kv.Value, text)) 
                                                    |> Async.AwaitTask 
                                  
                                        printfn "%b" send })

        |> Async.Parallel

    Async.RunSynchronously(asyncComp, cancellationToken=cts.Token)
    |> (fun _ -> fmdf.InputBlock.Complete())

