// Invoke the methods from incompatible types

type Cat() = 
    member this.Walk() = printfn "cat walk"
type Dog() = 
    member this.Walk() = printfn "dog walk"

let adapter() = 
    let cat = Cat()
    let dog = Dog()
    let inline walk (x : ^T) = (^T : (member Walk : unit->unit) (x))
    walk(cat)
    walk(dog)

adapter()