module SimpleAsyncExample

let printThenSleepThenPrint =
    async {
        printfn "before sleep"
        do! Async.Sleep 3000
        printfn "wake up"
    }
Async.StartImmediate printThenSleepThenPrint
printfn "continuing"

Async.Start printThenSleepThenPrint
printfn "continuing"

Async.RunSynchronously printThenSleepThenPrint
printfn "continuing"