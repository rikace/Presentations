// Learn more about F# at http://fsharp.net
module Program

open System
open CudaInterop
open GPUTranslation
open FSharp.Execution
open System.Reflection
open CudaInterop.InteropLibrary
open CUDARandom
open CUDABLAS
open CudaDataStructure

let blockIdx = new BlockDim()
let threadIdx = new ThreadIdx()

[<ReflectedDefinition; GPU>]
let sample (a:CUDAPointer2<float>) (b:CUDAPointer2<float>) (c:CUDAPointer2<float>)= 
    let x = blockIdx.x
    c.Set(a.[x] + b.[x], x) //c.[x] = a.[x] + b.[x]
    ()

[<ReflectedDefinition; GPU>]
let pascalTriangle (a:CUDAPointer2<float32>) (b:CUDAPointer2<float32>) =
    let x = blockIdx.x
    if x = 0 then
        b.Set(a.[x], x)
    else
        b.Set(a.[x] + a.[x-1], x)    
    ()

[<ReflectedDefinition; GPU>]
let bopm (a:CUDAPointer2<float32>) (b:CUDAPointer2<float32>) =
    let u = 0.2f
    let d = 1.f / u
    let x = blockIdx.x
    if x = 0 then
        b.Set(a.[x] * u, x)
    else
        b.Set(a.[x-1] * d, x)    
    ()

[<ReflectedDefinition; GPU>]
let sample2 (a:CUDAPointer2<float>) (b:CUDAPointer2<float>) (c:CUDAPointer2<float>)= 
    sample a b c
    let x = blockIdx.x
    for j=0 to x do
        c.Set(a.[j] + b.[j], x)
    let mutable i = 3
    while i >= 8 do
        if i > 3 then
            c.Set(a.[i] + b.[i], x)
        else
            c.Set(a.[i] + b.[0], x)
        i <- i + 1 

[<ReflectedDefinition; GPU>]
let sample4 (a:CUDAPointer2<float32>) (b:CUDAPointer2<float32>) : unit = 
    let x = blockIdx.x
    let mutable max = 0.f
    for i=x to 15 do
        if max < a.[i] then 
            max <- a.[i]
        else
            ()
    b.Set(max, x)

let tempFileName = @".\temp.cu"

let WriteToFile() = 
    let a1 = <@@ sample @@>    
    let b = getCUDACode(a1)
    let a2 = <@@ sample2 @@>
    let b2 = getCUDACode(a2)
    let a3 = <@@ pascalTriangle @@>
    let b3 = getCUDACode(a3)
    let commonCode = getCommonCode()

    System.IO.File.Delete(tempFileName)
    System.IO.File.WriteAllText(tempFileName, commonCode + b + b2 + b3);
    ()

let WriteToFile2() = 
    let a1 = <@@ bopm @@>    
    let b = getCUDACode(a1)
    let commonCode = getCommonCode()

    System.IO.File.Delete(tempFileName)
    System.IO.File.WriteAllText(tempFileName, commonCode + b);

let getMax() = 
    let tempFileName = @".\temp.cu"
    WriteToFile2()
    let execution = new GPUExecution()
    let m = execution.Init(tempFileName)
    let input = [1.f .. 15.f] |> Array.ofList
    let output = Array.create input.Length 0.f
    let r, ps = execution.Execute("sample4", [input; output;])
    let results = 
        ps 
        |> List.map (fun p -> p.ToArray() |> snd)
    ()

//getMax()

let input1 = [|1..5|]
let input2 = [|11..15|]
let out = Array.create 5 0.f

let test2() = 
    GenerateCodeToFile()    
        
    let execution = new GPUExecution()    
    
    let m = execution.Init(tempFile)
    let input1 = input1 |> Array.map float32
    let input2 = input2 |> Array.map float32
    let out = out |> Array.map float32
    let r, ps = execution.Execute("sample", [input1; input2; out])
    let results = 
        ps 
        |> List.map (fun p -> p.ToArray() |> snd)
    
    let random = CUDARandom.CUDARandom()
    let r, g = random.CreateGenerator( CUDARandomRngType.CURAND_PSEUDO_DEFAULT )
    let r, p = random.GenerateUniform(g, 5)
    let out = Array.create 5 0.f
    
    let out = execution.CopyDeviceToHost(CUDAPointer2(p), out)
    out |> Seq.iter (fun n -> printfn "%A" n)
    //let r, g = CUDARandom.CUDARandomDriver64.CreateGenerator( CUDARandomRngType.CURAND_PSEUDO_DEFAULT )
    //let r, p = CUDARandom.CUDARandomDriver64.GenerateUniform(g, 5)
    

    ()

let test1() = 
    GenerateCodeToFile()
    let execution = new GPUExecution()
    let m = execution.Init(tempFile)

    let input1 = input1 |> Array.map float32
    let input2 = input2 |> Array.map float32
    let out = out |> Array.map float32

    let r1, i1 = execution.Runtime.CopyHostToDevice(input1)
    let r2, i2 = execution.Runtime.CopyHostToDevice(input2)
    let l1 = execution.Runtime.GetSize(input1)
    let r3, i3 = execution.Runtime.CopyHostToDevice(out)      

    let r = execution.Runtime.ExecuteFunction("sample", [i1; i2; i3], input1.Length, 1)

    let result1, o1 = execution.Runtime.CopyDeviceToHost(i1, input1)
    let result2, o2 = execution.Runtime.CopyDeviceToHost(i2, input2)
    let result3, o3 = execution.Runtime.CopyDeviceToHost(i3, out)
    ()

let len = 100

let test3() =   
    GenerateCodeToFile() 
    let execution = new GPUExecution()
    let m = execution.Init(tempFile)
    //let len = 100

    let stopWatch =  System.Diagnostics.Stopwatch()
    stopWatch.Reset()
    stopWatch.Start()

    let l0 = Array.zeroCreate len
    let l1 = Array.zeroCreate len
    l0.[0] <- 1.f
    l1.[0] <- 0.f
    let r, p = execution.Runtime.CopyHostToDevice(l0)
    let r, p2 = execution.Runtime.CopyHostToDevice(l1)    
    let rs = 
        [1..len] 
        |> Seq.map (fun i ->
                        if i % 2 = 1 then
                            let r = execution.Runtime.ExecuteFunction("pascalTriangle", [p; p2], len, 1)
                            r
                        else 
                            let r = execution.Runtime.ExecuteFunction("pascalTriangle", [p2; p], len, 1)
                            r)
        |> Seq.toList

    let result1, o1 = execution.Runtime.CopyDeviceToHost(p, l0)
    let result2, o2 = execution.Runtime.CopyDeviceToHost(p2, l1)

    stopWatch.Stop()
    printfn "%A" stopWatch.Elapsed
    ()

let computePascal(p:float32 array, p2:float32 array) = 
    let len = p.Length
    [0..len-1] 
    |> Seq.iter (fun i -> 
                    if i = 0 then p2.[i] <- 1.f
                    else p2.[i] <- p.[i-1] + p.[i])

    ()

// normal pascal triagle
let test4() =
    let stopWatch =  System.Diagnostics.Stopwatch()
    stopWatch.Reset()
    stopWatch.Start()

    //let len = 100
    let l0 = Array.zeroCreate len
    let l1 = Array.zeroCreate len
    l0.[0] <- 1.f
    l1.[0] <- 0.f

    [1..len] 
    |> Seq.map (fun i ->
                        if i % 2 = 1 then
                            let r = computePascal(l0, l1)
                            r
                        else 
                            let r = computePascal(l1, l0)
                            r)
    |> Seq.toList
    |> ignore

    stopWatch.Stop()
    printfn "%A" stopWatch.Elapsed

    ()

let test5() = 
    let getIntPtr arr = 
        let nativeint = System.Runtime.InteropServices.Marshal.UnsafeAddrOfPinnedArrayElement(arr,0)
        let intptr = new System.IntPtr(nativeint.ToPointer())
        intptr
    let mutable ptr = IntPtr()
    let arr = [|1.f;2.f;3.f;4.f;5.f;1.f;2.f;3.f;4.f;5.f;|]
    let arr2 = [|11.f;12.f;13.f;14.f;15.f;11.f;12.f;13.f;14.f;15.f;|]
    let intptr = getIntPtr arr
    let intptr2 = getIntPtr arr2
    let size = arr.Length * sizeof<float>
    let error = CUDARuntime.CUDARuntime32.cudaMalloc(&ptr, SizeT(size))
    let error = CUDARuntime.CUDARuntime32.cudaMemcpy(ptr, intptr, SizeT(10*4), CUDAMemcpyKind.cudaMemcpyHostToDevice)
    //let error = CUDARuntime.CUDARuntime32.cudaMemset(&ptr, 1, size)
    let error = CUDARuntime.CUDARuntime32.cudaMemcpy(intptr2, ptr, SizeT(size), CUDAMemcpyKind.cudaMemcpyDeviceToHost)

    printfn "%A %A" arr arr2

let test6() = 
    let r = CUDARandom()
    let status, g = r.CreateGenerator(CUDARandomRngType.CURAND_PSEUDO_DEFAULT)
    if status = curandStatus.CURAND_SUCCESS then
        let status, v = r.GenerateNormal(g, 0, 0., 0.)
        if status = curandStatus.CURAND_SUCCESS then
            printfn "generated random value is = %A" v
        else
            printfn "generation failed. status = %A" status
    else
        printfn "create generator failed. status = %A" status
    ()

//test6()
//
//test5()
//test4()
//test3()
//test2()
//test1()

Console.ReadKey() |> ignore
