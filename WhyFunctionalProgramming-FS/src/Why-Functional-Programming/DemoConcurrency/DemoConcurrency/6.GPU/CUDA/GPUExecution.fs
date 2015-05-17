namespace FSharp.Execution

open System
open CudaInterop
open GPUTranslation
open System.Diagnostics
open ItemType
open System.IO
open Microsoft.FSharp.Quotations
open Microsoft.FSharp.Quotations.Patterns
open Microsoft.FSharp.Quotations.DerivedPatterns
open Microsoft.FSharp.Quotations.ExprShape
open CudaDataStructure
open CudaDataStructureExtensions

type BlockID() = 
    inherit dim3()

type GPUExecution () as this = 
    let runtime = new CUDARunTime()
    let nvcc = "nvcc.exe"
    do this.Init() |> ignore

    interface IDisposable with
        member this.Dispose() = ()    

    member this.Runtime with get() = runtime
    
    member private this.CompileToPTX() : string = 
        let fn = @".\ComputationLibrary.cu"
        this.CompileToPTX(fn)
    member private this.CompileToPTX(fn) : string = 
        let targetFile = System.IO.Path.GetFileNameWithoutExtension(fn) + ".ptx"
        System.IO.File.Delete(targetFile)
        use p = new Process()
        let para = sprintf "%s -ptx" fn
        p.StartInfo <- ProcessStartInfo(nvcc, para)
        p.StartInfo.UseShellExecute <- true
        p.StartInfo.WindowStyle <- ProcessWindowStyle.Hidden
        p.Start() |> ignore
        p.WaitForExit()
        targetFile
    member this.Init() = 
        let fn = this.CompileToPTX()
        let r, m = runtime.LoadModule(fn)
        if isSuccess r then m
        else failwith "cannot load moudule"

    member this.Init(fn:string) = 
        let fn = this.CompileToPTX(fn)
        let r,m = runtime.LoadModule(fn)
        if isSuccess r then m
        else failwith "cannot load module"

    member this.Execute(fn:string, list:'T array list) = 
        let unitSize = (sizeof<'T>) |> uint32
        let size = list.Head.Length |> uint32
        let results = 
            list 
            |> List.map (fun l -> this.Runtime.CopyHostToDevice(l))
            |> List.map (fun (r,p) -> (r, new CUDAArray<'T>(p, size, this.Runtime)))

        let success = results |> Seq.forall (fun (r,_) -> isSuccess(r))
        if success then
            let pointers = results |> List.map snd
            let head = List.head list
            let result = this.Runtime.ExecuteFunction(fn, pointers |> List.map box, head.Length, 1)
            let out = Array.create head.Length 0.f
            let a = this.Runtime.CopyDeviceToHost(pointers.[0].Pointer, out)
            (result, pointers)
        else
            failwith "copy host failed" 

    member this.CopyHostToDevice(data: 'T array) =
        let r, out = this.Runtime.CopyHostToDevice(data)
        if r = CUResult.Success then out
        else failwith "cannot copy host to device"

    member this.CopyDeviceToHost(p:CUDAPointer2<_>, data) =
        let r, out = this.Runtime.CopyDeviceToHost(p, data)
        if r = CUResult.Success then out
        else failwith "cannot copy device to host"

    member this.ToCUDAArray(l) = 
        let r, array = this.Runtime.CopyHostToDevice(l)
        if r = CUResult.Success then array
        else failwith "cannot copy host to device"

    member this.ExecuteFunction(fn:string, cudaArray:CUDAPointer list) =
        let r = this.Runtime.ExecuteFunction(fn, cudaArray |> List.map box, cudaArray.Length, 1)
        r