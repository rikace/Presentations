namespace RxPlaygroundFs

open System

type ValueOrEndOrError<'a> =
  | Value of 'a
  | End
  | Error of Exception

type MyEnumerable<'a> = {
  getEnumerator : unit -> MyEnumerator<'a>
}

and MyEnumerator<'a> = {
  next : unit -> ValueOrEndOrError<'a>
}

// flip the arrows

type AntiEnumerator<'a> = {
  //  next : unit -> ValueOrEndOrError<'a>
  antiNext : ValueOrEndOrError<'a> -> unit
}

module demo =
    let ae = {
      antiNext = function
                 | Value a -> ()
                 | End     -> ()
                 | Error e -> ()
    }

type AntiEnumerator'<'a> = {
  antiNext_Value : 'a -> unit
  antiNext_End   : unit -> unit
  antiNext_Error : Exception -> unit
}

type AntiEnumerable<'a> = {
  //  getEnumerator : unit -> MyEnumerator<'a>
  antiGetEnumerator : AntiEnumerator'<'a> -> unit
}

// renaming

type MyObserver<'a> = {
  onValue : 'a -> unit
  onEnd   : unit -> unit
  onError : Exception -> unit
}

type MyObservable<'a> = {
  setObserver : MyObserver<'a> -> unit
}

