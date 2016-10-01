open System
open SantaClausProblem

[<EntryPoint>]
let main argv = 
    //printfn "%A" argv
    let st = SantaClausProblem()
    let cancellationToken = st.Start()

    //cancellationToken.Cancel()

    Console.ReadLine() |> ignore
    0 // return an integer exit code



