module CudaDataStructureExtensions

open System
open CudaDataStructure
open CUDARuntime

let isSuccess r = 
    r = CUResult.Success

let isCudaSuccess r = 
    r = cudaError.cudaSuccess

let is64Bit = IntPtr.Size = 8

let newCudaPointer (length:int) =     
    let mutable intPtr = Unchecked.defaultof<IntPtr>
    let r = 
        if is64Bit then        
            CUDARuntime64.cudaMalloc(&intPtr, SizeT(length))
        else
            CUDARuntime32.cudaMalloc(&intPtr, SizeT(length))
    if not (isCudaSuccess r) then 
        failwith "cannot allocate"
    let p = CUDAPointer(intPtr)
    p