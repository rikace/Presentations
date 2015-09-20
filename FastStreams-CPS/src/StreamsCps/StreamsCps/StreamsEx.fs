namespace SimpleStreamsStopPushing

open System
open System.Collections.Generic

type Stream<'T> = ('T -> bool) -> unit

module StreamEx =

    let inline ofArray (source: 'T[]) : Stream<'T> =
        fun k ->
            let mutable i = 0
            let mutable next = true
            while i < source.Length && next do
                next <- k source.[i]
                i <- i + 1             

    let inline filter (predicate: 'T -> bool) (stream: Stream<'T>) : Stream<'T> =
        fun k -> stream (fun value -> if predicate value then k value else true)

    let inline map (mapF: 'T -> 'U) (stream: Stream<'T>) : Stream<'U> =
        fun k -> stream (fun v -> k (mapF v))

    let inline iter (iterF: 'T -> unit) (stream: Stream<'T>) : unit =
        stream (fun v -> iterF v; true)

    let inline toArray (stream: Stream<'T>) : 'T [] =
        let acc = new List<'T>()
        stream |> iter (fun v -> acc.Add(v))
        acc.ToArray()

    let inline fold (foldF: 'State -> 'T -> 'State) (state: 'State) (stream: Stream<'T>) : 'State =
        let acc = ref state
        stream (fun v -> acc := foldF !acc v; true)
        !acc

    let inline reduce (reducer: ^T -> ^T -> ^T) (stream: Stream< ^T >) : ^T
            when ^T : (static member Zero : ^T) =
        fold (fun s v -> reducer s v) LanguagePrimitives.GenericZero stream

    let inline sum (stream : Stream< ^T>) : ^T
            when ^T : (static member Zero : ^T)
            and ^T : (static member (+) : ^T * ^T -> ^T) =
        fold (+) LanguagePrimitives.GenericZero stream


    let inline takeWhile (predicate: 'T -> bool) (stream: Stream<'T>) : Stream<'T> =
        fun k -> stream (fun v -> if predicate v then k v else false)
