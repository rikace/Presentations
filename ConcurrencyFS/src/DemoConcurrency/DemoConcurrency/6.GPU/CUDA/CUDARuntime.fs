namespace CUDARuntime

open System
open System.Text
open System.Collections.Generic
open System.Runtime.InteropServices
open CudaDataStructure

module CUDARuntime64 = 
    [<Literal>]
    let dllName = "cudart64_60"
    [<DllImport(dllName)>]
    extern cudaError cudaMemcpy(IntPtr dst, IntPtr src, SizeT count, CUDAMemcpyKind kind);
    [<DllImport(dllName)>]
    extern cudaError cudaMalloc(IntPtr& p, SizeT size);
    [<DllImport(dllName)>]
    extern cudaError cudaMemset(IntPtr& p, int value, int count);

module CUDARuntime32 =     
    [<Literal>]
    let dllName = "cudart32_60"
    [<DllImport(dllName)>]
    extern cudaError cudaMemcpy(IntPtr dst, IntPtr src, SizeT count, CUDAMemcpyKind kind);
    [<DllImport(dllName)>]
    extern cudaError cudaMalloc(IntPtr& p, SizeT size);
    [<DllImport(dllName)>]
    extern cudaError cudaMemset(IntPtr& p, int value, int count);
