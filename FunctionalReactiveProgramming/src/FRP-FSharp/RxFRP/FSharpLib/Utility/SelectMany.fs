namespace Easj360FSharp 

open System

module SelectManyImp = 

    let selectMany1 (ab:'a -> 'b seq) (abc:'a -> 'b -> 'c) input =    
        input |> Seq.collect (fun a -> ab a |> Seq.map (fun b -> abc a b))

    let selectMany2 (source : 'TSource seq) (selector : 'TSource -> 'TResult seq) =    
        source |> Seq.collect selector
  
    let selectMany3 (source : 'TSource seq) (selector : 'TSource -> int -> 'TResult seq) =    
        source |> Seq.mapi (fun n s -> selector s n) |> Seq.concat
        
    let selectMany4 (source : 'TSource)                
                   (collectionSelector : 'TSource -> 'TCollection seq)               
                   (resultSelector : 'TSource -> 'TCollection -> 'TResult) =   
        source     
        |> Seq.collect (fun sourceItem ->    
                        collectionSelector sourceItem         
                        |> Seq.map (fun collection -> resultSelector sourceItem collection))
                        
    let selectMany5 (source : 'TSource)                
                   (collectionSelector : 'TSource -> int -> 'TCollection seq)               
                   (resultSelector : 'TSource -> 'TCollection -> 'TResult) =    
        source     
        |> Seq.mapi (fun n sourceItem ->         
                        collectionSelector sourceItem n        
                        |> Seq.map (fun collection -> resultSelector sourceItem collection))    
        |> Seq.concat

    let selectMany6 lst1 lst2 = lst1 |> List.collect (fun i1 -> lst2 |> List.map (fun i2 -> [i1, i2]))