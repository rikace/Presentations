// Factory pattern implementation: it returns different types based on inputs

type FactoryTypes =
	| TypeA
	| TypeA

type IA =
    abstract member Action : unit -> ()

let factorySample() = 
    let factory = function
      | TypeA -> { new IA with 
                       member this.Action() = printfn "type A" }
      | TypeB -> { new IA with 
                      member this.Action() = printfn "type B" }
    
    let output = factory Type.TypeA
    output.Action()

factorySample()