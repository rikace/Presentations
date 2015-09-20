module Adapter

//define a cat class
type Cat() = 
    member this.Walk() = printfn "cat walks"

// define a dog class
type Dog() = 
    member this.Walk() = printfn "dog walks"

// adapter pattern
let adapterExample() = 
    let cat = Cat()
    let dog = Dog()

    // define the GI function to invoke the Walk function
    let inline walk (x : ^T) = (^T : (member Walk : unit->unit) x)

    // invoke GI and both Cat and Dog
    walk(cat)
    walk(dog)
