#r "..\..\Lib\Microsoft.Accelerator.dll"
#load "..\Utilities\show.fs"
#r "FSharp.PowerPack.dll"

open System
open Microsoft.ParallelArrays

type A = Microsoft.ParallelArrays.ParallelArrays
type FPA = Microsoft.ParallelArrays.FloatParallelArray

(*
    This program sets up a pseudo-random two dimensional input array, 
    declares a two dimensional convolver and then uses two of the targets 
    supported by Accelerator to convolve the input array. The first target 
    uses a GPU to perform the convolution and the second target uses 
    multiple x64 processor cores and makes use of special SIMD instructions. 
 *)
let runAcceleratorTest() =
 
    // Declare a filter kernel for the convolution
    let testKernel = Array.map float32 [| 2; 5; 7; 4; 3 |]
 
    // Specify the size of each dimension of the input array
    let inputSize = 10
 
    // Create a pseudo-random number generator
    let random = Random (42)
   
    // Declare a psueduo-input data array
    let testData = Array2D.init inputSize inputSize (fun i j -> float32 (random.NextDouble() *
                                                                 float (random.Next(1, 100))))
   
    // Create an Accelerator float parallel array for the F# input array
    use testArray = new FloatParallelArray(testData)
 
    // Declare a function to convolve in the X or Y direction
    let rec convolve (shifts : int -> int []) (kernel : float32 []) i (a : FloatParallelArray)
       = let e = kernel.[i] * ParallelArrays.Shift(a, shifts i)
         if i = 0 then
           e
         else
           e + convolve shifts kernel (i-1) a
 
    // Declare a 2D convolver
    let convolveXY kernel input
       = // First convolve in the X direction and then in the Y direction
         let convolveX = convolve (fun i -> [| -i; 0 |]) kernel (kernel.Length - 1) input
         let convolveY = convolve (fun i -> [| 0; -i |]) kernel (kernel.Length - 1) convolveX
         convolveY
 
    // Create a DX9 target and use it to convolve the test input
    use dx9Target = new DX9Target()
    let convolveDX9 = dx9Target.ToArray2D (convolveXY testKernel testArray)
    printfn "DX9: -> \r\n%A" convolveDX9
    show (sprintf "DX9: -> \r\n%A" convolveDX9)
 
    // Create a X64 multi-core target and use it to convolve the test input
    use mcTarget = new MulticoreTarget()
    let convolveMC = mcTarget.ToArray2D (convolveXY testKernel testArray)
    printfn "MC: -> \r\n%A" convolveMC
    show (sprintf "MC: -> \r\n%A" convolveMC)

    show2 (sprintf "DX9: -> \r\n%A" convolveDX9) (sprintf "MC: -> \r\n%A" convolveMC)


runAcceleratorTest()

(* The program below shows how to implement this 1D convolution using 
    both the DX9 GPU target and the x64 SIMD multi-core target. *)

open System
open Microsoft.ParallelArrays
 
let runAcceleratorTestSingleConvolution() =
   
    // Declare an input data array
    let inputData = Array.map float32 [| 7; 2; 5; 9; 3; 8; 6; 4 |]
 
    // Declare filter kernel for the convolution
    let kernel = Array.map float32 [| 2; 5; 7; 4; 3 |]
    let kernelSize = Array.length kernel
 
    // Create an Accelerator float parallel array for the F# input array
    use inputArray = new FloatParallelArray(inputData)
 
    // Build the expression
    let rec expr i = let e = kernel.[i] *
                             ParallelArrays.Shift(inputArray, -i)
                     if i = 0 then
                       e
                     else
                       e + expr (i-1)
   
    // Create DX9 target and compute result
    use dx9Target = new DX9Target()
    let resDX = dx9Target.ToArray1D(expr (kernelSize-1))
    printfn "DX9   --> \r\n%A" resDX
    show (sprintf  "DX9   --> \r\n%A" resDX)

 
    // Create X64 multi-core target and compute result
    use mc64Target = new MulticoreTarget()
    let resMC = mc64Target.ToArray1D(expr (kernelSize-1))
    printfn "X64MC --> \r\n%A" resMC
    show (sprintf "X64MC --> \r\n%A" resMC)

 
runAcceleratorTestSingleConvolution()