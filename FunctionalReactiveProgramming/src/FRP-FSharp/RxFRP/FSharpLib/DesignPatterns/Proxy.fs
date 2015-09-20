module Proxy

// define core computation
type CoreComputation() = 
    member this.Add(x) = x + 1
    member this.Sub(x) = x - 1
    member this.GetProxy name = 
        match name with
        | "Add" -> this.Add, "add"
        | "Sub" -> this.Sub, "sub"
        | _ -> failwith "not supported"

// proxy implementation
let proxy() = 
    let core = CoreComputation()

    // get the proxy for the add function
    let proxy = core.GetProxy "Add"

    // get the compute from proxy
    let coreFunction = fst proxy

    // get the core function name
    let coreFunctionName = snd proxy

    // perform the core function calculation
    printfn "performed calculation %s and get result = %A" coreFunctionName (coreFunction 1)
