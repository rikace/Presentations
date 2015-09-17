open Scanning
open System
open System.Threading

[<EntryPoint>]
let main argv = 

    for id in 1 .. 10 do 
        let source = new CancellationTokenSource()
        runAgent.Post(createJob(id, source))

    printfn "Specify a job by number to cancel it, then press Enter." 

    let mutable finished = false 
    while not finished do 
        let input = System.Console.ReadLine()
        let a = ref 0
        if (Int32.TryParse(input, a) = true) then
            cancelJob(!a)
        else
            printfn "Terminating."
            finished <- true

    printfn "Type any key to continue.." 
    System.Console.ReadKey() |> ignore

    0 // return an integer exit code
