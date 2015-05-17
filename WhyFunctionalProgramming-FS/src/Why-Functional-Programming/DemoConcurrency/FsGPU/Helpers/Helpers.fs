namespace FsGPU

module Helpers =

    open System

    let iDivUp(a, b) =
        if ((a % b) <> 0) then (a / b + 1) else (a / b)

    let MeasureTime f arg =
        let start = DateTime.Now
        let res = f arg 
        let finish = DateTime.Now
        (res, TimeSpan(finish.Ticks - start.Ticks))

    let RepeatFunc n (f : 'a -> 'b) args =
        let mutable res = Unchecked.defaultof<'b>
        for i = 1 to n do 
            res <- f(args)
        res    

    let RunFunc n f title args =
        printfn "%s" title
        printf "%d times: " n
        let res, span = MeasureTime (RepeatFunc n f) args
        printfn "%d min, %d sec, %d ms" span.Minutes span.Seconds span.Milliseconds 
