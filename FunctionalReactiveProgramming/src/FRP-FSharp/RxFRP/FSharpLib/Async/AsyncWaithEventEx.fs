namespace Easj360FSharp 

#nowarn "40"
open System

module AsyncWaithEventEx = 
    // Implementation of the 'AwaitEvent' primitive

    type private Closure<'a>(f) =
      member x.Invoke(sender:obj, a:'a) : unit = f(a)
   
    type Microsoft.FSharp.Control.Async with 
      static member AwaitEvent(ev1:IEvent<'del, 'a>) = 
        Async.FromContinuations((fun (cont,econt,ccont) -> 
          let rec obj = new Closure<'a>(fun value ->
            ev1.RemoveHandler(del)
            cont(value) )
          and del = Delegate.CreateDelegate(typeof<'del>, obj, "Invoke") :?> 'del
          ev1.AddHandler(del)))

      static member AwaitEvent(ev1:IEvent<'del1, 'a>, ev2:IEvent<'del2, 'b>) = 
        Async.FromContinuations((fun (cont,econt,ccont) -> 
          let rec obj1 = new Closure<'a>(fun value ->
            ev1.RemoveHandler(del1)
            ev2.RemoveHandler(del2)
            cont(Choice1Of2(value)) )
          and obj2 = new Closure<'b>(fun value ->
            ev1.RemoveHandler(del1)
            ev2.RemoveHandler(del2)
            cont(Choice2Of2(value)) )
          and del1 = Delegate.CreateDelegate(typeof<'del1>, obj1, "Invoke") :?> 'del1
          and del2 = Delegate.CreateDelegate(typeof<'del2>, obj2, "Invoke") :?> 'del2
          ev1.AddHandler(del1)
          ev2.AddHandler(del2) ))

      static member AwaitEvent(ev1:IEvent<'del1, 'a>, ev2:IEvent<'del2, 'b>, ev3:IEvent<'del3, 'c>) = 
        Async.FromContinuations((fun (cont,econt,ccont) -> 
          let rec obj1 = new Closure<'a>(fun value ->
            ev1.RemoveHandler(del1)
            ev2.RemoveHandler(del2)
            ev3.RemoveHandler(del3)
            cont(Choice1Of3(value)) )
          and obj2 = new Closure<'b>(fun value ->
            ev1.RemoveHandler(del1)
            ev2.RemoveHandler(del2)
            ev3.RemoveHandler(del3)
            cont(Choice2Of3(value)) )
          and obj3 = new Closure<'c>(fun value ->
            ev1.RemoveHandler(del1)
            ev2.RemoveHandler(del2)
            ev3.RemoveHandler(del3)
            cont(Choice3Of3(value)) )
          and del1 = Delegate.CreateDelegate(typeof<'del1>, obj1, "Invoke") :?> 'del1
          and del2 = Delegate.CreateDelegate(typeof<'del2>, obj2, "Invoke") :?> 'del2
          and del3 = Delegate.CreateDelegate(typeof<'del3>, obj3, "Invoke") :?> 'del3
          ev1.AddHandler(del1)
          ev2.AddHandler(del2)
          ev3.AddHandler(del3) ))


