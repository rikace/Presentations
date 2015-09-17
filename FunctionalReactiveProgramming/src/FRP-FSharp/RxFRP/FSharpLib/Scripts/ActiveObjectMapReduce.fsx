#load "AgentSystem.fs"
open AgentSystem.LAgent
open System
open System.Collections
open System.Collections.Generic
open System.Threading

type IOutput<'out_key, 'out_value> =
    abstract Reduced: 'out_key -> seq<'out_value> -> unit
    abstract MapReduceDone: unit -> unit
    
type Mapper<'in_key, 'in_value, 'my_out_key, 'out_value when 'my_out_key : comparison>(map:'in_key -> 'in_value -> seq<'my_out_key * 'out_value>, i, partitionF) =
    let w = new WorkQueue()
    let mutable reducerTracker: BitArray = null
    let mutable controller = Unchecked.defaultof<Controller<'in_key, 'in_value, 'my_out_key, 'out_value>>
    let mutable reducers = Unchecked.defaultof<Reducer<'in_key, 'in_value, 'my_out_key, 'out_value> array>
    member m.Init c reds =
        w.Queue (fun () ->
            controller <- c
            reducers <- reds
            reducerTracker <- new BitArray(reducers.Length, false))
    member m.Process inKey inValue =
        w.Queue (fun () ->
            let outKeyValues = map inKey inValue
            outKeyValues |> Seq.iter (fun (outKey, outValue) ->
                                        let reducerUsed = partitionF outKey (reducers.Length)
                                        reducerTracker.Set(reducerUsed, true)
                                        reducers.[reducerUsed].Add(outKey, outValue)))
    member m.Done () =
        w.Queue (fun () ->
            controller.MapDone i reducerTracker)
    member m.Stop () = w.Stop ()            
    
and Reducer<'in_key, 'in_value, 'out_key, 'out_value when 'out_key : comparison>(reduce:'out_key -> seq<'out_value> -> seq<'out_value>, i, output:IOutput<'out_key, 'out_value>) =
    let w = new WorkQueue()
    let mutable workItems = new List<'out_key * 'out_value>()
    let mutable controller = Unchecked.defaultof<Controller<'in_key, 'in_value, 'out_key, 'out_value>>
    member r.Init c =
        w.Queue (fun () ->
            controller <- c)
    member r.StartReduction () =
        w.Queue (fun () ->
            workItems
            |> Seq.groupBy fst
            |> Seq.sortBy fst
            |> Seq.map (fun (key, values) -> (key, reduce key (values |> Seq.map snd)))
            |> Seq.iter (fun (key, value) -> output.Reduced key value)
            controller.ReductionDone i) 
    member r.Add (outKey:'out_key, outValue:'out_value) : unit =
        w.Queue (fun () ->
            workItems.Add((outKey, outValue)))
    member m.Stop () = w.Stop ()            
                
and Controller<'in_key, 'in_value, 'out_key, 'out_value when 'out_key : comparison>(output:IOutput<'out_key, 'out_value>) =
    let w = new WorkQueue()
    let mutable mapperTracker: BitArray = null
    let mutable reducerUsedByMappers: BitArray = null
    let mutable reducerDone: BitArray = null
    let mutable mappers = Unchecked.defaultof<Mapper<'in_key, 'in_value, 'out_key, 'out_value> array>
    let mutable reducers = Unchecked.defaultof<Reducer<'in_key, 'in_value, 'out_key, 'out_value> array>
    let BAtoSeq (b:BitArray) = [for x in b do yield x]
    member c.Init maps reds =
        w.Queue (fun () ->
            mappers <- maps
            reducers <- reds
            mapperTracker <- new BitArray(mappers.Length, false)
            reducerUsedByMappers <- new BitArray(reducers.Length, false)
            reducerDone <- new BitArray(reducers.Length, false))
    member c.MapDone (i : int) (reducerTracker : BitArray) : unit =
        w.Queue (fun () ->
            mapperTracker.Set(i, true)
            let reducerUsedByMappers = reducerUsedByMappers.Or(reducerTracker)
            if not( BAtoSeq mapperTracker |> Seq.exists(fun bit -> bit = false)) then
                BAtoSeq reducerUsedByMappers |> Seq.iteri (fun i r -> if r = true then reducers.[i].StartReduction ())
                mappers |> Seq.iter (fun m -> m.Stop ())
              )
    member c.ReductionDone (i: int) : unit =
        w.Queue (fun () ->
            reducerDone.Set(i, true)
            if BAtoSeq reducerDone |> Seq.forall2 (fun x y -> x = y) (BAtoSeq reducerUsedByMappers) then
                output.MapReduceDone ()
                reducers |> Seq.iter (fun r -> r.Stop ())
                c.Stop()
             )         
    member m.Stop () = w.Stop ()
                
let mapReduce   (inputs:seq<'in_key * 'in_value>)
                (map:'in_key -> 'in_value -> seq<'out_key * 'out_value>)
                (reduce:'out_key -> seq<'out_value> -> seq<'out_value>)
                (output:IOutput<'out_key, 'out_value>)
                M R partitionF =
                    
    let len = inputs |> Seq.length
    let M = if len < M then len else M
    
    let mappers = Array.init M (fun i -> new Mapper<'in_key, 'in_value, 'out_key, 'out_value>(map, i, partitionF))
    let reducers = Array.init R (fun i -> new Reducer<'in_key, 'in_value, 'out_key, 'out_value>(reduce, i, output))
    let controller = new Controller<'in_key, 'in_value, 'out_key, 'out_value>(output)
    
    mappers |> Array.iter (fun m -> m.Init controller reducers)
    reducers |> Array.iter (fun r -> r. Init controller )
    controller.Init mappers reducers
    
    inputs |> Seq.iteri (fun i (inKey, inValue) -> mappers.[i % M].Process inKey inValue)
    mappers |> Seq.iter (fun m -> m.Done ())    

let partitionF = fun key M -> abs(key.GetHashCode()) % M 

let map = fun (fileName:string) (fileContent:string) ->
            let l = new List<string * int>()
            let wordDelims = [|' ';',';';';'.';':';'?';'!';'(';')';'\n';'\t';'\f';'\r';'\b'|]
            fileContent.Split(wordDelims) |> Seq.iter (fun word -> l.Add((word, 1)))
            l :> seq<string * int>
                                  
let reduce = fun key (values:seq<int>) -> [values |> Seq.sum] |> seq<int>

let printer () =
  { new IOutput<string, int> with
        member o.Reduced key values = printfn "%A %A" key values
        member o.MapReduceDone () = printfn "All done!!"}
    
let testInput = ["File1", "I was going to the airport when I saw someone crossing"; "File2", "I was going home when I saw you coming toward me"]   
mapReduce testInput map reduce (printer ()) 2 2 partitionF

open System.IO
open System.Text

let gatherer(step) =
  let w = new WorkQueue()
  let data = new List<string * int>()
  let counter = ref 0 
  { new IOutput<string, int> with
        member o.Reduced key values =
            w.Queue (fun () ->
                if !counter % step = 0 then
                    printfn "Processed %i words. Now processing %s" !counter key 
                data.Add((key, values |> Seq.head))
                counter := !counter + 1)
        member o.MapReduceDone () =
            w.Queue (fun () ->
                data
                |> Seq.distinctBy (fun (key, _) -> key.ToLower())
                |> Seq.filter (fun (key, _) -> not(key = "" || key = "\"" || (fst (Double.TryParse(key)))))
                |> Seq.toArray
                |> Array.sortBy snd
                |> Array.rev
                |> Seq.take 20
                |> Seq.iter (fun (key, value) -> printfn "%A\t\t%A" key value)
                printfn "All done!!")           
        }
                        
let splitBook howManyBlocks fileName =
    let buffers = Array.init howManyBlocks (fun _ -> new StringBuilder())
    fileName
    |> File.ReadAllLines
    |> Array.iteri (fun i line -> buffers.[i % (howManyBlocks)].Append(line) |> ignore)
    buffers

let blocks1 = __SOURCE_DIRECTORY__ + "\kjv10.txt" |> splitBook 100
let blocks2 = __SOURCE_DIRECTORY__ + "\warandpeace.txt" |> splitBook 100
let input =
    blocks1
    |> Array.append blocks2
    |> Array.mapi (fun i b -> i.ToString(), b.ToString())
        
//mapReduce input map reduce (gatherer(1000)) 20 20 partitionF

type BookSplitter () =
    let blocks = new List<string * string>()
    member b.Split howManyBlocks fileName =
            let b =
                fileName
                |> splitBook howManyBlocks
                |> Array.mapi (fun i b -> i.ToString(), b.ToString())
            blocks.AddRange(b)
    member b.Blocks () =
            blocks.ToArray() :> seq<string * string>

type WordCounter () =
    let w = new WorkQueue()
    let words = new Dictionary<string,int>()
    let worker(wordCounter:WordCounter, ev:EventWaitHandle) =
          let w1 = new WorkQueue()
          { new IOutput<string, int> with
                member o.Reduced key values =
                    w1.Queue (fun() ->
                        wordCounter.AddWord key (values |> Seq.head))
                member o.MapReduceDone () =
                    w1.Queue(fun () ->
                        ev.Set() |> ignore)
           }
    member c.AddWord word count =
            let exist, value = words.TryGetValue(word)
            if exist then
                words.[word] <- value + count
            else
                words.Add(word, count)
    member c.Add fileName =
        w.Queue (fun () ->
            let s = new BookSplitter()
            fileName |> s.Split 100
            let ev = new EventWaitHandle(false, EventResetMode.AutoReset)
            let blocks = s.Blocks ()
            mapReduce blocks map reduce (worker(c, ev)) 20 20 partitionF
            ev.WaitOne() |> ignore
            )  
    member c.Words =
        w.QueueWithAsync (fun () ->
            words |> Seq.toArray |> Array.map (fun kv -> kv.Key, kv.Value)
        )    

let wc = new WordCounter()
wc.Add (__SOURCE_DIRECTORY__ + "\kjv10.txt")
wc.Add (__SOURCE_DIRECTORY__ + "\warandpeace.txt")

let wordsToPrint = async {
                    let! words = wc.Words
                    return words
                        |> Seq.distinctBy (fun (key, _) -> key.ToLower())
                        |> Seq.filter (fun (key, _) -> not(key = "" || key = "\"" || (fst (Double.TryParse(key)))))
                        |> Seq.toArray
                        |> Array.sortBy snd
                        |> Array.rev
                        |> Seq.take 20
                        |> Seq.iter (fun (key, value) -> printfn "%A\t\t%A" key value)}


Async.RunSynchronously wordsToPrint
                        
Thread.Sleep(15000)
printfn "Closed session"               