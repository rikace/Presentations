namespace utility

//#light

open System
open System.Numerics

module srcModule = 

    let maxIteration = 100

    let modSquared (c : Complex) = c.Real * c.Real + c.Imaginary * c.Imaginary

    type MandelbrotResult = 
        | DidNotEscape
        | Escaped of int
    
    let mandelbrot c =
        let rec mandelbrotInner z iterations =
            if(modSquared z >= 4.0) 
                then Escaped iterations
            elif iterations = maxIteration
                then DidNotEscape
            else mandelbrotInner ((z * z) + c) (iterations + 1)
        mandelbrotInner c 0

    let chars = " .:-;!/>)|&IH%*#"

    while true do 
        for y in [-1.2..0.05..1.2] do
            System.Threading.Thread.Sleep(300)
            for x in [-2.0..0.025..0.9] do
                match mandelbrot(Complex(x, y)) with
                | DidNotEscape -> Console.Write " "
                | Escaped i -> Console.Write chars.[i &&& 15]
            Console.WriteLine()     
