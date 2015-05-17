// Factory pattern implementation: it returns different types based on inputs

type FactoryTypes =
    | TypeA
    | TypeB

type IA =
    abstract member Action : unit -> unit

let factorySample() = 
    let factory = function
      | TypeA -> { new IA with 
                       member this.Action() = printfn "type A" }
      | TypeB -> { new IA with 
                      member this.Action() = printfn "type B" }
    
    let output = factory FactoryTypes.TypeA
    output.Action()

factorySample()



let makeCounter() =
    let localVal = ref 0
    let makeCounter'() =
        localVal := !localVal + 1
        !localVal 
    makeCounter'



let c1 = makeCounter()
c1()
c1()
c1()

let c2 = makeCounter()

printfn "C1 = %d, C2 = %d" (c1()) (c2())