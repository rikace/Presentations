namespace SimpleStreams

open System
open System.Collections.Generic

type Stream<'T> = ('T -> unit) -> unit

module Stream =

    let inline ofArray (source: 'T[]) : Stream<'T> =
        fun k ->
            let mutable i = 0
            while i < source.Length do
                k source.[i]
                i <- i + 1          

    // takeWhile : ('T -> bool) -> Stream<'T> -> Stream<'T>
    let inline takeWhile f stream =
        fun k -> stream (fun v -> if f v then k v else false)

    let inline filter (predicate: 'T -> bool) (stream: Stream<'T>) : Stream<'T> =
        fun k -> stream (fun value -> if predicate value then k value)

    let inline map (mapF: 'T -> 'U) (stream: Stream<'T>) : Stream<'U> =
        fun k -> stream (fun v -> k (mapF v))

    let inline iter (iterF: 'T -> unit) (stream: Stream<'T>) : unit =
        stream (fun v -> iterF v)

    let inline toArray (stream: Stream<'T>) : 'T [] =
        let acc = new List<'T>()
        stream |> iter (fun v -> acc.Add(v))
        acc.ToArray()

    let inline fold (foldF: 'State -> 'T -> 'State) (state: 'State) (stream: Stream<'T>) : 'State =
        let acc = ref state
        stream (fun v -> acc := foldF !acc v)
        !acc

    let inline reduce (reducer: ^T -> ^T -> ^T) (stream: Stream< ^T >) : ^T
            when ^T : (static member Zero : ^T) =
        fold (fun s v -> reducer s v) LanguagePrimitives.GenericZero stream

    let inline sum (stream : Stream< ^T>) : ^T
            when ^T : (static member Zero : ^T)
            and ^T : (static member (+) : ^T * ^T -> ^T) =
        fold (+) LanguagePrimitives.GenericZero stream
