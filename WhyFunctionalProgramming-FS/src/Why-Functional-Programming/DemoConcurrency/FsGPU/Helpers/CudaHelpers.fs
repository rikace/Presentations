namespace FsGPU

open Microsoft.FSharp.Control
open System
open System.Threading
open GASS.CUDA
open GASS.CUDA.Types
open System.IO
open System.Collections

module CudaHelpers =
    
    let GetDeviceCount() =
        let mutable count = ref 0
        let cuda = new CUDA(true)
        let count = cuda.Devices.Length 
        cuda.Dispose()
        count 
