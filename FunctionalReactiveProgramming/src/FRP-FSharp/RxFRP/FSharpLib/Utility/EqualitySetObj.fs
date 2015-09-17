module EqualitySetObj

open System
open System.IO

    // class can be used to convert LIST to Set and remove duplicates
    type TestComp<'T when 'T : equality>(value:'T,  equalFuncf:'T -> 'T -> bool, compareFunc: 'T -> 'T -> int) =  
    // when 'T : equality and 'T :> System.IComparable>(value:'T, equalFuncf:'T -> 'T -> bool, compareFunc: 'T -> 'T -> int) = 
        member x.Value with get() = value

        override x.Equals(y) =
            match y with
            :? TestComp<'T> as z -> equalFuncf x.Value z.Value
            | _ -> false

        override x.GetHashCode() =
            value.GetHashCode()

        interface System.IComparable with
            member x.CompareTo y =
                match y with
                :? TestComp<'T> as z -> compareFunc x.Value z.Value
                | _ -> invalidArg "" ""
        

//    let rec getFiles dir (filter:FileInfo -> bool) = seq {
//        for file in Directory.EnumerateFiles(dir, "*.*", SearchOption.TopDirectoryOnly) do 
//                let fileInfo = System.IO.FileInfo(file)
//                if (filter fileInfo) then yield fileInfo
//        for dir' in Directory.EnumerateDirectories(dir, "*.*", SearchOption.TopDirectoryOnly) do yield! getFiles dir' filter }
//
//    let s1 = seq { 1..3 }
//    let s2 = seq { 2..7 }
//    let s3 = seq { 5..9 }
//
//    let l1 = [1..40000]
//    let l2 = [3..10000]
//
//    let s4 = s1 |> Seq.map (fun f -> TestComp(f, (fun x y -> x = y), (fun x y -> x.CompareTo(y))))
//    let s5 = s2 |> Seq.map (fun f -> TestComp(f, (fun x y -> x = y), (fun x y -> x.CompareTo(y))))
//    let s6 = s3 |> Seq.map (fun f -> TestComp(f, (fun x y -> x = y), (fun x y -> x.CompareTo(y))))
//
//    let s = [s1; s2; s3]
//    let ss = [s4; s5; s6]
//    
//    let append2 a b =
//            let rec loop acc = function
//                | [] -> acc
//                | x::xs -> loop (x::acc) xs
//            loop b a
//
//    let append3 a b =
//            let rec loop acc = function
//                | [] -> acc
//                | x::xs -> if List.exists (fun elem -> x = elem) acc then loop acc xs
//                           else loop (x::acc) xs
//            loop b a
//
//    let append4 a b =
//        let rec append = function
//            | cont, [], ys -> cont ys
//            | cont, x::xs, ys -> append ((fun acc -> cont (x::acc)), xs, ys)
//        append(id, a, b)
//
//    let r2 = [l1; l2] |> List.reduce(fun acc elem -> List.append acc elem) |> Set.ofSeq
//
//
//   // let r1 = s |> List.reduce(fun acc elem -> (Set. (Set.ofSeq acc) (Set.ofSeq elem)) |> Set.toSeq) //|> Set.ofSeq
//   
////    let r2 = s |> List.reduce(fun acc elem -> Seq.append acc elem) |> Set.ofSeq
//
//    
//    Seq.iter(fun f -> printfn "%A" f) r2
//
//    let dirs = [@"c:\temp\Output"; @"c:\temp\Output1"; @"c:\temp\Output2"]
//
//    let fileUnifier (dirs:string list) (filter:System.IO.FileInfo -> bool) =
//        let (a, b) =
//            dirs
//            |> List.map(fun dir -> getFiles dir filter)
//            |> Seq.concat
//            |> Seq.groupBy(fun f -> (f.Extension, f.Length))
//            |> Seq.toList
//            |> List.partition (fun (_, files) -> Seq.length files > 1)
//        0
//
//        //|> List.reduce(fun acc elem -> Seq.append acc elem)// |> Set.ofSeq
//        
//    let res = fileUnifier dirs (fun f -> true)
//        
////        (fun s -> TestComp(FileInfo(s), (fun f1 f2 -> f1.Length = f2.Length), (fun f1 f2 -> f1.Length.CompareTo(f2.Length)))) f)   //System.IO.FileInfo(s)) f)
////        |> List.reduce(fun acc elem -> Seq.append acc elem) 
////        |> Set.ofSeq
////
////    fileUnifier dirs
