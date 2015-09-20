namespace SimpleStreams


module ObservableEx =

   open System
   let inline ofSeq (xs:'T seq) : IObservable<'T> =
      { new IObservable<'T> with
         member __.Subscribe(observer) =
            for x in xs do observer.OnNext(x)
            observer.OnCompleted()
            { new IDisposable with member __.Dispose() = ()}           
      }

   let inline sum (observable:IObservable< ^T >) : IObservable< ^T >
         when ^T : (static member ( + ) : ^T * ^T -> ^T) 
         and  ^T : (static member Zero : ^T) = 
      { new IObservable<'T> with 
         member this.Subscribe(observer:IObserver<'T>) =
            let acc = ref (LanguagePrimitives.GenericZero)
            let accumulator =
               { new IObserver<'T> with 
                  member __.OnNext(x) = acc := !acc + x
                  member __.OnCompleted() = observer.OnNext(!acc)
                  member __.OnError(_) = failwith "Not implemented"
               }
            observable.Subscribe(accumulator)
      }

   let inline first (observable:IObservable<'T>) : 'T =
      let value = ref (Unchecked.defaultof<'T>)
      let _ = observable.Subscribe(fun x -> value := x)
      !value
