namespace FsFRPx

open System

 module Misc =

 

  let catOption l =
    let rec proc l acc =
        match l with
        |[] -> acc
        |h::t -> match h with
                 |Some x -> proc t (x::acc)
                 |None -> proc t acc
    List.rev (proc l [])
   
  let isSome x =  match x with 
                  |None -> false
                  |Some _ -> true

  let getSome x =
    match x with
    |Some x -> x
    |None -> failwith "error: not some"
             
  let curry f a b = f (a,b)
  let uncurry f (a, b) = f a b