#light

namespace FsReactive

 open FsReactive
 
 module Integration = 
 
 // aliasB : 'a -> ('a Behavior * ('a -> unit))

  let aliasB x0 =
    let xref = ref x0
    let rec bb = Beh (fun _ -> (!xref, fun()->bb))
    (bb, fun x -> xref := x)
  
// bindAliasB : 'a Behavior -> ('b Behavior * ('a -> unit)) -> 'a Behavior

  let bindAliasB xb aliasB =
    let rec bf aliasedB t = let (Beh bbf) = aliasedB
                            let (Beh bbf') = fst aliasB
                            let (r,nb) = bbf t
                            (snd aliasB) r
                            let (r',_) = bbf' t
                            (r, (fun() -> Beh (bf (nb()))))
    memoB (Beh (bf xb))
    
  type NumClass<'a, 'b> = 
    {plus : 'a -> 'a -> 'a;
     minus : 'a -> 'a -> 'a;
     mult : 'a -> 'b -> 'a;
     div : 'a -> 'b -> 'a;
     neg : 'a -> 'a
    }
    
  let floatNumClass = 
    { plus = (+);
      minus = (-);
      mult = (*);
      div = (/);
      neg = (fun (x:float) -> -x)
    }
  
 // integrateGenB : NumClass<'a, Time> -> 'a Behavior -> Time -> 'a -> 'a Behavior

  let integrateGenB numClass b t0 i0 = 
    let rec bf b t0 i0 t = let (r,nb) = atB b t
                           let i = numClass.plus i0 (numClass.mult r (t-t0))
                           (i, fun() -> Beh (bf (nb()) t i))
    Beh (bf b t0 i0 )   
     
// integrate : float Behavior -> Time -> float -> float Behavior

  let integrate b t0 i0 = integrateGenB floatNumClass b t0 i0


 // integrateGenB : NumClass<'a, Time> -> ('a -> 'a) Behavior -> 'a Behavior -> Time -> 'a -> 'a Behavior
  
  let integrateWithConstraintsGenB numClass constraintsBf b t0 i0 = 
    let rec bf constraintsBf b t0 i0 t = 
                           let (r,nb) = atB b t
                           let i = numClass.plus i0 (numClass.mult r (t-t0))
                           let (rcf, ncB) = atB constraintsBf t
                           let i' = rcf i
                           (i', fun() -> Beh (bf (ncB()) (nb()) t i'))
    Beh (bf constraintsBf b t0 i0 )   
    

 // integrateWithConstraints :  (float -> float) Behavior -> float Behavior -> Time -> float -> float Behavior
  
  let integrateWithConstraints b t0 i0 = integrateWithConstraintsGenB floatNumClass b t0 i0


