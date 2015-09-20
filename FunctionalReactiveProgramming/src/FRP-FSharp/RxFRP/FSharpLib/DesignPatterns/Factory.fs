module Factory

// define the interface
type IA = 
  abstract Action : unit -> unit

// define two types
type Type = 
  | TypeA 
  | TypeB

let factoryPattern() = 
    // factory pattern to create the object according to the input object type
    let factory = function
      | TypeA -> { new IA with 
                       member this.Action() = printfn "I am type A" }
      | TypeB -> { new IA with 
                       member this.Action() = printfn "I am type B" }

    // create type A object
    let obj1 = factory TypeA
    obj1.Action()

    // create type B object
    let obj2 = factory TypeB
    obj2.Action()
