module AsyncAndParallel

let parallel2 (job1, job2) =
    async {
        let! task1 = Async.StartChild job1
        let! task2 = Async.StartChild job2
        let! res1 = task1
        let! res2 = task2
        return (res1, res2) }

let printThenSleepThenPrint x =
    async {
        printfn "before sleep %A" x
        do! Async.Sleep 3000
        printfn "wake up %A" x
        return x
    }

let printResult =
    async {
        let! res = parallel2 (printThenSleepThenPrint 1, printThenSleepThenPrint System.DateTime.Now)
        printf "%A" res
    }

Async.Start printResult
