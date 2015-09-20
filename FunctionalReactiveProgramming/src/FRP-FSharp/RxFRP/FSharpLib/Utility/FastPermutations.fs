namespace Easj360FSharp


module FastPermutations =
    // From: http://stackoverflow.com/questions/286427/calculating-permutations-in-f
    // Much faster than anything else I've tested

    let rec insertions x = function
        | []             -> [[x]]
        | (y :: ys) as l -> (x::l)::(List.map (fun x -> y::x) (insertions x ys))

    let rec permutations = function
        | []      -> seq [ [] ]
        | x :: xs -> Seq.concat (Seq.map (insertions x) (permutations xs))


module GeneralPermutations =

    let rec permutations (A : 'a list) =
            if List.isEmpty A then [[]] else
            [
                for a in A do
                yield! A |> List.filter (fun x -> x <> a) 
                         |> permutations
                         |> List.map (fun xs -> a::xs)
            ]


    let append (a:string) (lst:string list) = 
        lst |> List.collect (fun ch -> if a.Contains(ch) then [] else [a+ch])

    let generateCombination (lst:string list) = 
        let total = lst.Length
        let rec combination (acc:string list list) (src:string list) (target: string list) = 
            let result = [for i in src -> target |> append i] |> List.collect id
            match result.Head.Length with
            | x when x = total -> result::acc
            | _ -> combination (result::acc) result target

        combination [lst] lst lst
  
    for str in generateCombination [for i in "abc" -> i.ToString()] |> List.rev |> List.collect id do
        str |> System.Console.WriteLine 

// Input: "ab"
// output: "a" "b" "ab" "ba"


