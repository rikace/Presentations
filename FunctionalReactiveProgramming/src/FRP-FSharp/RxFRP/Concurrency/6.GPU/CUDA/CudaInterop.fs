namespace CudaInterop

open System
open System.Text
open System.Collections.Generic
open System.Runtime.InteropServices
open CudaDataStructure

module CUDA32 = 
    [<DllImport("cudart32_60")>]
    extern cudaError cudaGetDeviceProperties(CUDADeviceProp& prop, int device)
module CUDA64 = 
    [<DllImport("cudart64_60")>]
    extern cudaError cudaGetDeviceProperties(CUDADeviceProp& prop, int device)

module InteropLibrary = 
    [<DllImport("nvcuda")>]
    extern CUResult cuModuleLoad( CUDAModule& m, string fn)
    [<DllImport("nvcuda")>]
    extern CUResult cuDriverGetVersion( int& driverVersion);
    [<DllImport("nvcuda")>]
    extern CUResult cuInit(uint Flags);
    [<DllImport("nvcuda", EntryPoint = "cuCtxCreate_v2")>]
    extern CUResult cuCtxCreate(CUDAContext& pctx, uint flags, CUDADevice dev);
    [<DllImport("nvcuda")>]
    extern CUResult cuDeviceGet(CUDADevice& device, int ordinal)
    [<DllImport("nvcuda")>]
    extern CUResult cuModuleGetFunction(CUDAFunction& hfunc, CUDAModule hmod, string name)
    [<DllImport("nvcuda")>]
    extern CUResult cuFuncSetBlockShape(CUDAFunction hfunc, int x, int y, int z);
    [<DllImport("nvcuda")>]
    extern CUResult cuLaunch(CUDAFunction f)
    [<DllImport("nvcuda")>]
    extern CUResult cuLaunchGrid(CUDAFunction f, int grid_width, int grid_height)

    [<DllImport("nvcuda", EntryPoint = "cuMemAlloc_v2")>]
    extern CUResult cuMemAlloc(CUDAPointer& dptr, uint bytesize);
    [<DllImport("nvcuda", EntryPoint = "cuMemcpyDtoH_v2")>]
    extern CUResult cuMemcpyDtoH(IntPtr dstHost, CUDAPointer srcDevice, uint ByteCount)
    [<DllImport("nvcuda", EntryPoint = "cuMemcpyHtoD_v2")>]
    extern CUResult cuMemcpyHtoD(CUDAPointer dstDevice, IntPtr srcHost, uint ByteCount)
    [<DllImport("nvcuda", EntryPoint = "cuMemFree_v2")>]
    extern CUResult cuMemFree(CUDAPointer dptr)

    [<DllImport("nvcuda")>]
    extern CUResult cuParamSeti(CUDAFunction hfunc, int offset, uint value)
    [<DllImport("nvcuda")>]
    extern CUResult cuParamSetf(CUDAFunction hfunc, int offset, float32 value)
    [<DllImport("nvcuda")>]
    extern CUResult cuParamSetv(CUDAFunction hfunc, int offset, int64& value, uint numbytes) 
    [<DllImport("nvcuda")>]
    extern CUResult cuParamSetSize(CUDAFunction hfunc, uint numbytes)

    [<DllImport("nvcuda", EntryPoint = "cuMemsetD8_v2")>]
    extern CUResult cuMemsetD8(CUDAPointer dstDevice, byte uc, uint N)
    [<DllImport("nvcuda", EntryPoint = "cuMemsetD16_v2")>]
    extern CUResult cuMemsetD16(CUDAPointer dstDevice, uint16 us, uint N);
  
module InteropLibrary2 = 
    [<DllImport("nvcuda")>]
    extern CUResult cuParamSetv(CUDAFunction hfunc, int offset, IntPtr ptr, uint numbytes)

type CUDADriver() = 
    member private this.Is64Bit
        with get() = IntPtr.Size = 8

    member this.CUDAProperty 
        with get() = 
            let mutable property = CUDADeviceProp()
            let returnCode = if this.Is64Bit then  CUDA64.cudaGetDeviceProperties(&property, 0)
                             else CUDA32.cudaGetDeviceProperties(&property, 0)
            property
    member this.Version 
        with get() = 
            let mutable version = 0
            (InteropLibrary.cuDriverGetVersion(&version), version)

type CUDAArray<'T>(cudaPointer:CUDAPointer2<_>, size:uint, runtime:CUDARunTime) =
    let unitSize = uint32(sizeof<'T>) 
    interface IDisposable with 
        member this.Dispose() = runtime.Free(cudaPointer) |> ignore
    member this.Runtime with get() = runtime
    member this.SizeInByte with get() = size
    member this.Pointer with get() = cudaPointer
    member this.UnitSize with get() = unitSize
    member this.Size with get() = int( this.SizeInByte / this.UnitSize )
    member this.ToArray<'T>() = 
        let out = Array.create (int(size)) Unchecked.defaultof<'T>
        this.Runtime.CopyDeviceToHost(this.Pointer, out)
               
and CUDARunTime(deviceID) = 
    let mutable device = CUDADevice()
    let mutable deviceContext = CUDAContext()
    let mutable m = CUDAModule()    

    let init() = 
        let r = InteropLibrary.cuInit(deviceID)
        let r = InteropLibrary.cuDeviceGet(&device, int(deviceID))
        let r = InteropLibrary.cuCtxCreate(&deviceContext, deviceID, device)
        ()
    do init()

    let align(offset, alignment) = offset + alignment - 1 &&& ~~~(alignment - 1);
    new() = new CUDARunTime(0u)
   
    interface IDisposable with
        member this.Dispose() = ()

    member this.LoadModule(fn) =
        (InteropLibrary.cuModuleLoad(&m, fn), m)
    member this.Version
        with get() = 
            let mutable a = 0
            (InteropLibrary.cuDriverGetVersion(&a), a)
    member this.Is64Bit with get() = CudaDataStructureExtensions.is64Bit
    member this.GetFunction(fn) = 
        let mutable f = CUDAFunction()
        (InteropLibrary.cuModuleGetFunction(&f, m, fn), f)
    member this.ExecuteFunction(fn, x, y) =
        let r, f = this.GetFunction(fn)
        if r = CUResult.Success then
            InteropLibrary.cuLaunchGrid(f, x, y)
        else
            r
    member this.ExecuteFunction(fn) = 
        let r, f = this.GetFunction(fn)
        if r = CUResult.Success then
            InteropLibrary.cuLaunch(f)
        else
            r
    member this.ExecuteFunction(fn, [<ParamArray>] parameters:obj list) = 
        let func = this.GetFunctionPointer(fn)
        this.SetParameter(func, parameters) 
        let r = InteropLibrary.cuLaunch(func)
        r
    member this.ExecuteFunction(fn, parameters:obj list, x, y) = 
        let func = this.GetFunctionPointer(fn)
        let paras = 
            parameters
            |> List.map (fun n -> match n with 
                                  | :? CUDAPointer2<float> as p -> box(p.Pointer)
                                  | :? CUDAPointer2<float32> as p -> box(p.Pointer)
                                  | :? CUDAPointer2<_> as p -> box(p.Pointer)
                                  | _ -> n)

        this.SetParameter(func, paras) 
        InteropLibrary.cuLaunchGrid(func, x, y)
    member private this.GetFunctionPointer(fn) = 
        let r, p = this.GetFunction(fn)
        if r = CUResult.Success then p
        else failwith "cannot get function pointer"

    // allocate
    member this.Allocate(bytes:uint) =
        let mutable p = CUDAPointer()
        (InteropLibrary.cuMemAlloc(&p, bytes), CUDAPointer2(p))
    member this.Allocate(array) = 
        let size = this.GetSize(array) |> uint32
        this.Allocate(size)
    member this.GetSize(data:'T array) = 
        this.MSizeOf(typeof<'T>) * uint32(data.Length)
    member this.GetUnitSize(data:'T array) = 
        this.MSizeOf(typeof<'T>)
    member private this.MSizeOf(t:Type) =
        if t = typeof<System.Char> then 2u
        else Marshal.SizeOf(t) |> uint32
    member this.Free(p:CUDAPointer2<_>) : CUResult = 
        InteropLibrary.cuMemFree(p.Pointer)

    member this.CopyHostToDevice(data: 'T array) =
        let gCHandle = GCHandle.Alloc(data, GCHandleType.Pinned)
        let size = this.GetSize(data)
        let r, p = this.Allocate(size)
        let r = (InteropLibrary.cuMemcpyHtoD(p.Pointer, gCHandle.AddrOfPinnedObject(), size), p)
        gCHandle.Free()
        r
    member this.CopyDeviceToHost(p:CUDAPointer2<_>, data) =
        let gCHandle = GCHandle.Alloc(data, GCHandleType.Pinned)
        let r = (InteropLibrary.cuMemcpyDtoH(gCHandle.AddrOfPinnedObject(), p.Pointer, this.GetSize(data)), data)
        gCHandle.Free()
        r   

    //parameter setting
    member private this.SetParameter<'T>(func, offset, vector:'T) = 
        let gCHandle = GCHandle.Alloc(vector, GCHandleType.Pinned)
        let numbytes = uint32(Marshal.SizeOf(vector))
        let r = InteropLibrary2.cuParamSetv(func, offset, gCHandle.AddrOfPinnedObject(), numbytes)
        gCHandle.Free()
        r
    member private this.SetParameterSize(func, size) = 
        if InteropLibrary.cuParamSetSize(func, size) = CUResult.Success then ()
        else failwith "set parameter size failed"
    member this.SetParameter(func, parameters) = 
        let mutable num = 0
        for para in parameters do
            match box(para) with
            | :? uint32 as n -> 
                num <- align(num, 4)
                if InteropLibrary.cuParamSeti(func, num, n) = CUResult.Success then ()
                else failwith "set uint32 failed"
                num <- num + 4
            | :? float32 as f ->
                num <- align(num, 4)
                if InteropLibrary.cuParamSetf(func, num, f) = CUResult.Success then ()
                else failwith "set float failed"
                num <- num + 4
            | :? int64 as i64 ->
                num <- align(num, 8)
                let mutable i64Ref = i64
                if InteropLibrary.cuParamSetv(func, num, &i64Ref, 8u) = CUResult.Success then ()
                else failwith "set int64 failed"
                num <- num + 8
            | :? char as ch ->
                num <- align(num, 2)
                let bytes = Encoding.Unicode.GetBytes([|ch|])
                let v = BitConverter.ToUInt16(bytes, 0)
                if this.SetParameter(func, num, v) = CUResult.Success then ()
                else failwith "set char failed"
                num <- num + 2            
            | :? CUDAPointer as devPointer ->
                num <- align(num, devPointer.PointerSize)
                if devPointer.PointerSize = 8 then
                    if this.SetParameter(func, num, uint64(int64(devPointer.Pointer))) = CUResult.Success then ()
                    else failwith "set device pointer failed"
                else
                    if InteropLibrary.cuParamSeti(func, num, uint32(int(devPointer.Pointer))) = CUResult.Success then ()
                    else failwith "set device pointer failed"
                num <- num + devPointer.PointerSize
            | :? CUDAArray<float32> as devArray ->
                let devPointer:CUDAPointer2<_> = devArray.Pointer
                num <- align(num, devPointer.PointerSize)
                if devPointer.PointerSize = 8 then
                    if this.SetParameter(func, num, uint64(int64(devPointer.Pointer.Pointer))) = CUResult.Success then ()
                    else failwith "set device pointer failed"
                else
                    if InteropLibrary.cuParamSeti(func, num, uint32(int(devPointer.Pointer.Pointer))) = CUResult.Success then ()
                    else failwith "set device pointer failed"
                num <- num + devPointer.PointerSize
            | _ when para.GetType().IsValueType ->
                let n = int(this.MSizeOf(para.GetType()))
                num <- align(num, n)
                if this.SetParameter(func, num, box(para)) = CUResult.Success then ()
                else failwith "set no-char object"
                num <- num + n
            | _ -> failwith "not supported"
        this.SetParameterSize( func, uint32(num) )
