namespace Easj360FSharp
    
    open System
    open System.Threading
    
    module Parallel2 =

        let parallel2 (job1, job2) =
            async { 
                    let! task1 = Async.StartChild job1
                    let! task2 = Async.StartChild job2
                    let! res1 = task1
                    let! res2 = task2
                    return (res1, res2) }

        let startChild (job1) =
            async { 
                    let! task1 = Async.StartChild job1
                    return! task1
                    }
                    
