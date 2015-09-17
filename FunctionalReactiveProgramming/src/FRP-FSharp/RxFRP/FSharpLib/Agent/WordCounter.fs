namespace Easj360FSharp

open System
open System.IO
open System.Threading
open System.ComponentModel

module WordCounter =

/// The internal type of messages for the agent
type Message = Word of string 
               | Fetch of AsyncReplyChannel<Map<string,int>> 
               | Stop
 
type WordCountingAgent() = 
    let counter = MailboxProcessor.Start(fun inbox ->
             // The states of the message processing state machine...
             let rec loop(words : Map<string,int>) = 
                async { let! msg = inbox.Receive()
                        match msg with 
                        | Word word ->
                            if words.ContainsKey word then
                                let count = words.[word]
                                let words = words.Remove word
                                return! loop(words.Add (word, (count + 1)) )
                            else
                                // do printfn "New word: %s" word
                                return! loop(words.Add (word, 1) )
                                
                        | Stop -> 
                            // exit
                            return ()
                        | Fetch  replyChannel  -> 
                            // post response to reply channel and continue
                            do replyChannel.Reply(words) 
                            return! loop(words) } 
 
             // The initial state of the message processing state machine...
             loop(Map.empty))
 
    member a.AddWord(n) = counter.Post(Word(n))
    member a.Stop() = counter.Post(Stop)
    member a.Fetch() = counter.PostAndAsyncReply(fun replyChannel -> Fetch(replyChannel))
let counter = new WordCountingAgent()
 
let readLines file = 
  seq { use r = new StreamReader( File.OpenRead file ) 
        while not r.EndOfStream do yield r.ReadLine() } 
 
let processFile file =
    let lines = readLines file
    for line in lines do
        let punctuation = [| ' '; '.'; '"'; ''';  ','; ';'; ':'; '!'; '?'; '-'; '('; ')'; |]
        let words = line.Split(punctuation)
        for word in words do
            if word.Length > 0 then
                counter.AddWord word
 
let printWords = false
 
//let main() =
//    let autoResetEvent = new AutoResetEvent(false)
//    let files = Directory.GetFiles(@"C:\Users\robert\Documents\Fielding")
//    let i = ref 0
//    for file in files do
//        use readfile = new BackgroundWorker()
//        readfile.DoWork.Add(fun _ ->
//            printfn "Starting '%s'" (Path.GetFileNameWithoutExtension file)
//            processFile file |> ignore )
//        readfile.RunWorkerCompleted.Add(fun _ -> 
//            printfn "Finished '%s'" (Path.GetFileNameWithoutExtension file)
//            incr i
//            if !i = files.Length then
//                autoResetEvent.Set() |> ignore)
//        readfile.RunWorkerAsync()
//    while not (autoResetEvent.WaitOne(100, false)) do
//        let words = counter.Fetch()
//        printfn "Words: %i" words.Count
//    let res = counter.Fetch()
//    
//    printfn "Finished Words: %i" res.Count
//    if printWords then
//        res.Iterate (fun k v -> printfn "%s : %i" k v)
//    counter.Stop()
//    read_line()
//    
//main()
// 
////
//   while not (autoResetEvent.WaitOne(100, false)) do
//        let words = counter.Fetch()
//        printfn "Words: %i" words.Count
// 
type Agent<'a> = MailboxProcessor<'a>
 
type Summation =
    | Add of int
    | Total of AsyncReplyChannel<int>    
and SummationAgent () =
    let agent = Agent.Start ( fun inbox ->    
        let rec loop total =
            async {
            let! message = inbox.Receive()
            match message with 
            | Add n -> do! loop (n + total)
            | Total reply -> reply.Reply(total)
            }
        loop 0
    )    
    /// Adds value to total
    member this.Add n = Add(n) |> agent.Post
    /// Returns total and ends computation
    member this.Total () = (fun reply -> Total(reply)) |> agent.PostAndReply
 
/// Invokes specified function with numbers from 1 to limit
let numberSource f limit =
    async {
        for i = 1 to limit do
            f i            
            if i % 10 = 0 then System.Console.WriteLine("{0}\t({1})",i,limit)        
    }
 
do  /// Summation agent instance
    let agent = SummationAgent ()    
    // Post series of numbers to summation agent in parallel
    [100;50;200]
    |> Seq.map (numberSource agent.Add)
    |> Async.Parallel
    |> Async.RunSynchronously
    |> ignore    
    // Get total
    let value = agent.Total ()
    System.Diagnostics.Debug.Assert(26425 = value);
    value |> System.Console.WriteLine

