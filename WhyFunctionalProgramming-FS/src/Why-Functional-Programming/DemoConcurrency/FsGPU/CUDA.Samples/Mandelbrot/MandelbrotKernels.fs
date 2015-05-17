(* Copyright 2009 FsGPU Project.
 *
 * Contributors to this file:
 * Alex Slesarenko - http://slesarenko.blogspot.com
 *
 * This file is part of FsGPU.  FsGPU is licensed under the 
 * GNU Library General Public License (LGPL). See License.txt for a complete copy of the
 * license.
 *)

namespace FsGPU.Cuda.Samples

open Microsoft.FSharp.Control
open System
open System.Threading
open GASS.CUDA
open GASS.CUDA.Engine
open GASS.CUDA.Types
open System.IO
open FsGPU.Cuda
open FsGPU.Helpers

module MandelbrotKernels = 
    
    let threadsX, threadsY = 32,16   (* only 512 threads per block allowed *)
    
    (* GPU execution code for mandelbrot calculation.
      NOTE: This method is not thread safe, so you need to call it from appropriate thread. (see CudaFunc class) *)
    let MandelbrotExecution = fun (ctx, (width : int, height : int, crunch : int, xOff : float32, yOff: float32, scale : float32)) ->
        let exec = ctx.Execution                                                   
            
        let num = width * height
        let colors = Array.zeroCreate<int> num 
        exec.AddParameter("colors", colors, ParameterDirection.Out) |> ignore
        exec.AddParameter("width", uint32(width)) |> ignore
        exec.AddParameter("height", uint32(height)) |> ignore
        exec.AddParameter("crunch", uint32(crunch)) |> ignore
        exec.AddParameter("xOff", float32(xOff)) |> ignore
        exec.AddParameter("yOff", float32(yOff)) |> ignore
        exec.AddParameter("scale", float32(scale)) |> ignore
        
        let blocksX, blocksY = iDivUp(width, threadsX), iDivUp(height, threadsY)
        exec.Launch(blocksX, blocksY, threadsX, threadsY, 1) |> ignore // Mandelbrot0_sm10<<<blocksX, blocksY, threadsX, threadsY>>>(dcolors,...);
        
        exec.ReadData(colors, 0)
        colors
    
    (* Calculate mandelbrot set for specified parameters using GPU kernel execution. 
    /// NOTE: this is thread-safe invoker of corresponding GPU execution  *)
    let CalcMandelbrot =
        let executor = new CudaExecutor<_, int[]>(0, "Mandelbrot_sm10.cubin", "Mandelbrot0_sm10", MandelbrotExecution)
        executor.GetInvoker()     
    



