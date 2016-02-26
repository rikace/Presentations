
namespace Common

 module Random = 

    open System
    open Vector

    let rand = new System.Random();
    let randAngle() = rand.NextDouble() * 2.0*Math.PI    
    let randFloat() = 2.0 * rand.NextDouble() - 1.0
    let random() = rand.NextDouble()
    let randRange min max = min + rand.NextDouble() * (max-min)
    let randVector() = Vector(randFloat(), randFloat())
    

