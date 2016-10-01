module Lists

open System
open System.IO
open CommonHelpers
open Microsoft.FSharp.Collections 
open System.Collections.Generic
open System.Linq
open FSharpx.Collections



let iterartions = 5

// Array    
module ArrayTest =
    
 
    let start() =     
        let words = ParallelizingFuzzyMatch.Data.Words
        let wordwordTuple = Array.Parallel.init words.Length (fun i -> (i, words.[i]))

        let item = wordwordTuple.Last()


        BenchPerformance.Time("Array map", (fun () -> 
            let map = 
                wordwordTuple
                |> Array.map(fun (i,m) -> i+1, m.ToLower())
            let map = map
            ()), iterations=iterartions ,startFresh=true)

        BenchPerformance.Time("Array map parallel", (fun () -> 
            let map = 
                wordwordTuple
                |> Array.Parallel.map(fun (i,m) -> i+1, m.ToLower())
            let map = map
            ()), iterations=iterartions ,startFresh=true)

        BenchPerformance.Time("Array filter", (fun () -> 
            let map = 
                wordwordTuple
                |> Array.filter(fun (i,m) -> i % 2 = 0)
            let map = map
            ()), iterations=iterartions ,startFresh=true)

        BenchPerformance.Time("Array filter parallel", (fun () -> 
            let map = 
                wordwordTuple
                |> Array.Parallel.choose(fun (i,m) -> if i % 2 = 0 then Some() else None)
            let map = map
            ()), iterations=iterartions ,startFresh=true)

        BenchPerformance.Time("Array Lookup", (fun () -> 
            let map = 
                wordwordTuple
                |> Array.find(fun (i,m) -> m = (snd item))
            let map = map
            ()), iterations=iterartions ,startFresh=true)

        BenchPerformance.Time("Array sort", (fun () -> 
            let map = 
                wordwordTuple
                |> Array.sort 
            let map = map
            ()), iterations=iterartions ,startFresh=true)

        BenchPerformance.Time("Array sort in place", (fun () -> 
            let map = 
                wordwordTuple
                |> Array.sortInPlaceBy(fun (m,i) -> m) 
            let map = map
            ()), iterations=iterartions ,startFresh=true)

module ResizeArrayTest =
    
 
    let start() =     
        let words = ParallelizingFuzzyMatch.Data.Words
        let wordwordTuple = Array.Parallel.init words.Length (fun i -> (i, words.[i]))

        let item = wordwordTuple.Last()


        BenchPerformance.Time("ResizeArray Insert bulk", (fun () -> 
            let arr = new ResizeArray<int*string>(wordwordTuple)
            let map = arr
            ()), iterations=iterartions ,startFresh=true)

        let arr = new ResizeArray<int*string>(wordwordTuple)

        BenchPerformance.Time("ResizeArray map", (fun () -> 
            let map = 
                arr
                |> Seq.map(fun (i,m) -> i+1, m.ToLower())
            let map = map |> Seq.length
            ()), iterations=iterartions ,startFresh=true)

      
        BenchPerformance.Time("ResizeArray filter", (fun () -> 
            let map = 
                arr
                |> Seq.filter(fun (i,m) -> i % 2 = 0)
            let map = map |> Seq.length
            ()), iterations=iterartions ,startFresh=true)


        BenchPerformance.Time("Array Lookup", (fun () -> 
            let map = 
                arr
                |> Seq.find(fun (i,m) -> m = (snd item))
            let map = map 
            ()), iterations=iterartions ,startFresh=true)

        BenchPerformance.Time("ResizeArray sort", (fun () -> 
            let map = 
                arr
                |> Seq.sort 
            let map = map
            ()), iterations=iterartions ,startFresh=true)

        BenchPerformance.Time("ResizeArray sort in place", (fun () -> 
            
            arr.Sort()            
            let map = arr
            ()), iterations=iterartions ,startFresh=true)

module ListTest =
    
 
    let start() =     
        let words = ParallelizingFuzzyMatch.Data.Words
        let wordwordTuple = Array.Parallel.init words.Length (fun i -> (i, words.[i]))

        let item = wordwordTuple.Last()


        BenchPerformance.Time("List Insert bulk", (fun () -> 
            let arr = wordwordTuple |> Array.toList
            let map = arr
            ()), iterations=iterartions ,startFresh=true)

        let arr = wordwordTuple |> Array.toList

        BenchPerformance.Time("List map", (fun () -> 
            let map = 
                arr
                |> List.map(fun (i,m) -> i+1, m.ToLower())
            let map = map |> List.length
            ()), iterations=iterartions ,startFresh=true)

      
        BenchPerformance.Time("List filter", (fun () -> 
            let map = 
                arr
                |> List.filter(fun (i,m) -> i % 2 = 0)
            let map = map |> List.length
            ()), iterations=iterartions ,startFresh=true)


        BenchPerformance.Time("List Lookup", (fun () -> 
            let map = 
                arr
                |> List.find(fun (i,m) -> m = (snd item))
            let map = map 
            ()), iterations=iterartions ,startFresh=true)

        BenchPerformance.Time("List sort", (fun () -> 
            let map = 
                arr
                |> List.sort
            let map = map
            ()), iterations=iterartions ,startFresh=true)

        BenchPerformance.Time("ResizeArray sort in place", (fun () -> 
            
            let arr = arr |> List.sortBy(fun i -> snd i)         
            let map = arr
            ()), startFresh=true)



module SeqTest =
    
 
    let start() =     
        let words = ParallelizingFuzzyMatch.Data.Words
        let wordwordTuple = Array.Parallel.init words.Length (fun i -> (i, words.[i]))

        let item = wordwordTuple.Last()


        BenchPerformance.Time("Seq Insert bulk", (fun () -> 
            let arr = wordwordTuple |> Array.toSeq
            let map = arr 
            ()), iterations=iterartions ,startFresh=true)
        let arr = wordwordTuple |> Array.toList

        BenchPerformance.Time("Seq map", (fun () -> 
            let map = 
                arr
                |> Seq.map(fun (i,m) -> i+1, m.ToLower())
            let map = map |> Seq.length
            ()), iterations=iterartions ,startFresh=true)
      
        BenchPerformance.Time("Seq filter", (fun () -> 
            let map = 
                arr
                |> Seq.filter(fun (i,m) -> i % 2 = 0)
            let map = map |> Seq.length
            ()), iterations=iterartions ,startFresh=true)

        BenchPerformance.Time("Seq Lookup", (fun () -> 
            let map = 
                arr
                |> Seq.find(fun (i,m) -> m = (snd item))
            let map = map 
            ()), iterations=iterartions ,startFresh=true)
        BenchPerformance.Time("Seq sort", (fun () -> 
            let map = 
                arr
                |> Seq.sort
            let map = map
            ()), iterations=iterartions ,startFresh=true)
        BenchPerformance.Time("Seq sort in place", (fun () -> 
            
            let arr = arr |> Seq.sortBy(fun i -> snd i)         
            let map = arr
            ()), iterations=iterartions ,startFresh=true)
module SetTest =
    
 
    let start() =     
        let words = ParallelizingFuzzyMatch.Data.Words
        let wordwordTuple = Array.Parallel.init words.Length (fun i -> (i, words.[i]))

        let item = wordwordTuple.Last()


        BenchPerformance.Time("Set Insert bulk", (fun () -> 
            let arr = wordwordTuple |> Array.toSeq |> Set
            let map = arr 
            ()), iterations=iterartions ,startFresh=true)
        let arr = wordwordTuple |> Array.toSeq |> Set

        BenchPerformance.Time("Set map", (fun () -> 
            let map = 
                arr
                |> Set.map(fun (i,m) -> i+1, m.ToLower())
            let map = map |> Set.count
            ()), iterations=iterartions ,startFresh=true)
      
        BenchPerformance.Time("Set filter", (fun () -> 
            let map = 
                arr
                |> Set.filter(fun (i,m) -> i % 2 = 0)
            let map = map |> Set.count  
            ()), iterations=iterartions ,startFresh=true)

        BenchPerformance.Time("Set Lookup", (fun () -> 
            let map = 
                arr
                |> Set.filter(fun (i,m) -> m = (snd item))
            let map = map.First()
            ()), iterations=iterartions ,startFresh=true)
        
module HashTest =
    
 
    let start() =     
        let words = ParallelizingFuzzyMatch.Data.Words
        let wordwordTuple = Array.Parallel.init words.Length (fun i -> (i, words.[i]))

        let item = wordwordTuple.Last()


        BenchPerformance.Time("Hash Insert bulk", (fun () -> 
            let arr = new HashSet<int*string>(wordwordTuple)
            let map = arr 
            ()), iterations=iterartions ,startFresh=true)
        let arr = new HashSet<int*string>(wordwordTuple)

        BenchPerformance.Time("Hash map", (fun () -> 
            let map = 
                arr
                |> Seq.map(fun (i,m) -> i+1, m.ToLower())
            let map = map |> Seq.length
            ()), iterations=iterartions ,startFresh=true)
      
        BenchPerformance.Time("Hash filter", (fun () -> 
            let map = 
                arr
                |> Seq.filter(fun (i,m) -> i % 2 = 0)
            let map = map |> Seq.length
            ()), iterations=iterartions ,startFresh=true)

        BenchPerformance.Time("Hash Lookup", (fun () -> 
            let map = 
                arr
                |> Seq.find(fun (i,m) -> m = (snd item))
            let map = map 
            ()), iterations=iterartions ,startFresh=true)
        BenchPerformance.Time("Hash sort", (fun () -> 
            let map = 
                arr
                |> Seq.sort
            let map = map
            ()), iterations=iterartions ,startFresh=true)
        BenchPerformance.Time("Hash sort in place", (fun () -> 
            
            let arr = arr |> Seq.sortBy(fun i -> snd i)         
            let map = arr
            ()), iterations=iterartions ,startFresh=true)


module DicTest = 
        let start() =     
        let words = ParallelizingFuzzyMatch.Data.Words
        let wordwordTuple = Array.Parallel.init words.Length (fun i -> (i, words.[i]))

        let item = wordwordTuple.Last()

        //let arr = new HashSet<int*string>(wordwordTuple)

        
        BenchPerformance.Time("Dict Insert bulk", (fun () -> 
            let arr = wordwordTuple |> dict
            let map = arr 
            ()), iterations=iterartions ,startFresh=true)

        let arr = wordwordTuple |> dict

        BenchPerformance.Time("Dict Map", (fun () -> 
            let map = 
                arr 
                |> Seq.map(fun i -> (i.Key + 1), (i.Value.ToLower()))
            let arr = map |> Seq.length
            ()), iterations=iterartions ,startFresh=true)

        BenchPerformance.Time("Dict filter", (fun () -> 
          
            let map = arr |> Seq.filter(fun i -> i.Key % 2 = 0)
            let arr = map |> Seq.length
            ()), iterations=iterartions ,startFresh=true)

        BenchPerformance.Time("Dict Lookup", (fun () -> 
            let map = arr |> Seq.filter(fun i -> i.Value = snd item)
            ()), iterations=iterartions ,startFresh=true)


module MapTest = 
        let start() =     
        let words = ParallelizingFuzzyMatch.Data.Words
        let wordwordTuple = Array.Parallel.init words.Length (fun i -> (i, words.[i]))

        let item = wordwordTuple.Last()

        //let arr = new HashSet<int*string>(wordwordTuple)

        
        BenchPerformance.Time("Map Insert bulk", (fun () -> 
            let arr = wordwordTuple |> Map.ofSeq
            let map = arr 
            ()), iterations=iterartions ,startFresh=true)

        BenchPerformance.Time("Map add", (fun () -> 
            let rec add lst (map:Map<int,string>) =
                match lst with
                | [] -> map
                | i::h -> map.Add i

            let map = add (wordwordTuple |> Array.toList) Map.empty
            ()), iterations=iterartions ,startFresh=true)


        let arr = wordwordTuple |> Map.ofSeq

        BenchPerformance.Time("Map Map", (fun () -> 
            let map = 
                arr 
                |> Map.map(fun i s-> (i + 1), (s.ToLower()))
            let arr = map |>Seq.length
            ()), iterations=iterartions ,startFresh=true)

        BenchPerformance.Time("Map filter", (fun () -> 
          
            let map = arr |> Map.filter(fun i _ -> i % 2 = 0)
            let arr = map |> Seq.length
            ()), iterations=iterartions ,startFresh=true)

        BenchPerformance.Time("Map Lookup", (fun () -> 
            let map = arr |> Map.findKey(fun i s -> s = snd item)
            ()), iterations=iterartions ,startFresh=true)
module HashMapTest = 
        let start() =     
        let words = ParallelizingFuzzyMatch.Data.Words
        let wordwordTuple = Array.Parallel.init words.Length (fun i -> (i, words.[i]))

        let item = wordwordTuple.Last()

        //let arr = new HashSet<int*string>(wordwordTuple)

        
        BenchPerformance.Time("Hash Map Insert bulk", (fun () -> 
            let arr = wordwordTuple |> PersistentHashMap.ofSeq
            let map = arr 
            ()), iterations=iterartions ,startFresh=true)

        BenchPerformance.Time("Hash Map add", (fun () -> 
            let rec add lst (map:PersistentHashMap<int,string>) =
                match lst with
                | [] -> map
                | i::h -> map.Add i

            let map = add (wordwordTuple |> Array.toList) PersistentHashMap.empty
            ()), iterations=iterartions ,startFresh=true)



        let arr = wordwordTuple |> PersistentHashMap.ofSeq


        BenchPerformance.Time("Hash Map Map", (fun () -> 
            let map = 
                arr 
                |> Seq.map(fun (s,i)-> (s+1),(i.ToLower()))
            let arr = map |>Seq.length
            ()), iterations=iterartions ,startFresh=true)

        BenchPerformance.Time("Hash Map filter", (fun () -> 
          
            let map = arr |> Seq.filter(fun (i,s) -> i % 2 = 0)
            let arr = map |> Seq.length
            ()), iterations=iterartions ,startFresh=true)

        BenchPerformance.Time("Hash Map Lookup", (fun () -> 
            let map = arr |> Seq.find(fun (i,s) -> s = snd item)
            ()), iterations=iterartions ,startFresh=true)


// HashSet

// converting arrra to seq to list to hash 

//HashMultiMap Initial Data Scaling Performance
//Internal.Utilities.Collections
// FSharpx.Collections.PersistentHashMap


let table = PersistentHashMap.empty<string, int>   //.Empty //() //<string, int>() //(HashIdentity.Structural)            

let t1 = table.Add("a", 1).Add("b", 2).Add("c", 3)