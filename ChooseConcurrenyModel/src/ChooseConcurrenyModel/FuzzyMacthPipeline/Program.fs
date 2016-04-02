

[<EntryPoint>]
let main argv = 
    
    PipeLineFuzzyMatchModule.loadDataAsyncInParalallAndProcess() |> ignore

    System.Console.ReadLine() |> ignore


    PipeLineFuzzyMatchModule.cts.Cancel()
    printfn "Cancelled!!"

    System.Console.ReadLine() |> ignore


    0 // return an integer exit code
