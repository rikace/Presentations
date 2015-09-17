module Singleton

type A private () =
    static let instance = A()
    static member Instance = instance
    member this.Action() = printfn "action from type A"

// singleton pattern
let singletonPattern() = 
    let a = A.Instance
    a.Action()
