#light


namespace FsReactive

 open FsReactive
 open Misc
 
 module DynCol = 
 
 // 'a option Behavior list -> ['a option Behavior] Event -> 'a list Behavior

  let rec dyncolB lb evt = 
    let isSome x =  match x with 
                    |None -> false
                    |Some _ -> true
    let bf t = let (r,nb) = List.unzip (List.filter (fst >> isSome) 
                                                    (List.map (fun b -> atB b t) lb ))
               let proc () = let l = List.map (fun x -> x()) nb
                             let (re, ne) = atE evt t
                             let l' = match re with
                                      |None -> l
                                      |Some b -> b@l
                             dyncolB l' (ne())
               (catOption r, proc)       
    Beh bf

    

    
    
       
              
              
              
         