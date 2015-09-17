module PrivateDataClass

type Circle = { 
    Radius : float; 
    X : float; 
    Y : float } 
with
    member this.Area = this.Radius**2. * System.Math.PI
    member this.Diameter = this.Radius * 2.
let myCircle = {Radius = 10.0; X = 5.0; Y = 4.5}
printfn "Area: %f Diameter: %f" myCircle.Area myCircle.Diameter
