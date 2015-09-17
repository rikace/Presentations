namespace Easj360FSharp 

open System
open System.Threading

module Permutations =
    
    let rec comb n lst = 
        let rec findChoices = function 
          | h::t -> (h,t) :: [ for (x,l) in findChoices t -> (x,l) ] 
          | []   -> [] 
        [ if n=0 then yield [] else 
                for (e,r) in findChoices lst do 
                    for o in comb (n-1) r do yield e::o  ] 

    let rec insertions x = function 
        | []             -> [[x]] 
        | (y :: ys) as l -> (x::l)::(List.map (fun x -> y::x) (insertions x ys)) 
 
    let rec permutations = function 
        | []      -> seq [ [] ] 
        | x :: xs -> Seq.concat (Seq.map (insertions x) (permutations xs)) 

      //F# for Scientists 
    let rec distribute e = function 
          | [] -> [[e]] 
          | x::xs' as xs -> (e::xs)::[for xs in distribute e xs' -> x::xs] 
 
    let rec permute = function 
          | [] -> [[]] 
          | e::xs -> List.collect (distribute e) (permute xs) 

    let permuteXS xs =
        let insert e xs =
            List.fold (fun (a,l) x -> (e::x::l)::List.map (fun ys -> x::ys) a,x::l) ( [ [ e ] ] , [ ] ) xs |> fst  
        let rec _p xs sofar =
            match xs with 
                | [] -> sofar
                | h::t -> 
                    _p t (List.fold (fun a x -> List.fold (fun s x -> x::s) a (insert h x)) sofar sofar)
        _p xs [[]] |> List.sort |> List.tail
        
    
    let rec orderedSubsets l =
             let rec insert x xs =
                  match xs with
                    | [] -> [[x]]
                    | y::ys -> (x::y::ys)::List.map (fun u -> y::u) (insert x ys)
             match l with
             | [] -> [[]]
             | x::xs -> let subs = orderedSubsets xs
                        subs |> List.collect (insert x) |> List.append subs

    let permuteList xs =
          let insert e xs =
            List.fold (fun (a,l) x -> (e::x::l)::List.map (fun ys -> x::ys) a,x::l) ( [ [ e ] ] , [ ] ) xs |> fst  
          let rec _p xs sofar =
                match xs with 
                | [] -> sofar
                | h::t -> 
                  _p t (List.fold (fun a x -> List.fold (fun s x -> x::s) a (insert h x)) sofar sofar)
          _p xs [[]] |> List.sort |> List.tail

    let runTest f =
        System.Diagnostics.Trace.Write("Start New Test")
        let s = System.Diagnostics.Stopwatch.StartNew()
        let res = f ["apple";"banana"; "canteloupe"; "ananas";"orange";"cherry"]        
//        res |> List.iter(fun x -> x 
//                               |> List.map (fun g -> g+ " - ") 
//                               |> List.iter (fun g -> System.Diagnostics.Trace.Write(g)))
        //|> ignore
        do System.Diagnostics.Trace.WriteLine(s.ElapsedMilliseconds)
        do System.Diagnostics.Trace.WriteLine("====================")
        res

    let convertToSeq res = 
        res 
        |> List.map (fun l -> l |> List.toSeq)
        |> List.toSeq

    let testPermuteList() =
        let res = runTest permuteList
        res 
        |> convertToSeq

    let testorderedSubsets() =
        let res = runTest orderedSubsets
        res 
        |> convertToSeq

    let testpermuteXS() =
        let res = runTest permuteXS
        res 
        |> convertToSeq

//    let areAllEqual() = 
//        let r1 = testPermuteList()
//        let r2 = testorderedSubsets()
//        System.Diagnostics.Trace.WriteLine(r1.Length)
//        System.Diagnostics.Trace.WriteLine(r2.Length)
//        r1 = r2.Tail
        