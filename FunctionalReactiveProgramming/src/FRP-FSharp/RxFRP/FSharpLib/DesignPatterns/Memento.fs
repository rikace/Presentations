module Memento

// define location record
type Location = { X : float; Y : float }

// define a particle class with a location property
type Particle() = 
    let mutable loc = {X = 0.; Y = 0.}
    member this.Loc 
        with get() = loc
        and private set v = loc <- v
    member this.GetMemento() = this.Loc
    member this.Restore v = this.Loc <- v
    member this.MoveXY(newX, newY) = loc <- { X = newX; Y = newY }


// create a particle
let particle = Particle()

// save current state
let currentState = particle.GetMemento()
printfn "current location is %A" particle.Loc

// move particle to new location
particle.MoveXY(2., 3.)
printfn "current location is %A" particle.Loc

// restore particle to previous saved location
particle.Restore currentState
printfn "current location is %A" particle.Loc
