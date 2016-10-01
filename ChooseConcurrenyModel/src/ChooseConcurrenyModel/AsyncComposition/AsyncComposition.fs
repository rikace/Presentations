module AsyncCompositionModule

open System
open System.IO
open System.Net
open System.Threading.Tasks
open ParallelizingFuzzyMatch
open FuzzyMatch

let urls = Data.GetTextFileUrls()

let loadData () =         
        seq { for url in urls do
                use client = new WebClient()
                let headerText = "Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.1; Trident/6.0)"
                client.Headers.Add("user-agent", headerText)
                yield client.DownloadString(url.Value) }




let loadAsync (url) : Async<string> = async {
    use client = new WebClient()
    let headerText = "Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.1; Trident/6.0)"
    client.Headers.Add("user-agent", headerText)
    return! client.AsyncDownloadString(Uri(url)) }




let loadDataAsyncInParalall() : Async<string []>=
    urls
    |> Seq.map(fun kv -> loadAsync kv.Value)
    |> Async.Parallel


let splitWords (text:string) =
    text.Split([|'\n'|])
    |> Array.map(fun lines -> lines.Split(Data.Delimiters))    
    |> Array.concat

let processFuzzyMatch words =
    Data.WordsToSearch
    |> Seq.map(fun input -> JaroWinkler.Parallel.bestMatch words (input.ToUpper()))
    |> Seq.concat
    |> Seq.map(fun result -> result.Word)

let composeProcessFuzzyMatch = splitWords >> processFuzzyMatch




let loadDataAsyncInParalallAndProcess() =
    urls
    |> Seq.map(fun kv -> async {    let! text = loadAsync kv.Value
                                    return composeProcessFuzzyMatch text })
    |> Async.Parallel
    |> Async.RunSynchronously
    |> Seq.concat


//
//
//let loadDataAsyncWorkerInParalallAndProcess() =
//    let jobs =     
//        urls
//        |> Seq.map(fun kv -> async {    let! text = loadAsync kv.Value
//                                        return composeProcessFuzzyMatch text }) 
//    
//    let worker = AsyncWorker.AsyncWorker<_>(jobs)
//    worker.ProgressChanged.Add(fun result -> ())


// worker.JobCompleted.Add(fun (jobNumber, result) -> 

//        printfn "job %d completed with result %A" jobNumber result.Length)
//
//    worker.AllCompleted.Add(fun results -> 
//        printfn "all done, results = %A" results )
//
//    worker.Start()
