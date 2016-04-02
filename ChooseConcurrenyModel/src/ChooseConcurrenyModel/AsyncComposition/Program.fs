
open CommonHelpers

[<EntryPoint>]
let main argv = 

    BenchPerformance.Time("Load Data Async Paralall and Process", fun () ->
        let matches = AsyncCompositionModule.loadDataAsyncInParalallAndProcess()
        for m in matches do 
            printf "%s\t" m )

    System.Console.ReadKey() |> ignore

    0 // return an integer exit code
