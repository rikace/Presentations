namespace CUDARandom

open System
open System.Text
open System.Collections.Generic
open System.Runtime.InteropServices
open CudaDataStructure
open CUDARuntime

type curandStatus = 
    | CURAND_SUCCESS = 0
    | CURAND_VERSION_MISMATCH = 100
    | CURAND_NOT_INITIALIZED = 101
    | CURAND_ALLOCATION_FAILED = 102
    | CURAND_TYPE_ERROR =103
    | CURAND_OUT_OF_RANGE = 104
    | CURAND_LENGTH_NOT_MULTIPLE = 105
    | CURAND_LAUNCH_FAILURE = 201
    | CURAND_PREEXISTING_FAILURE = 202
    | CURAND_INITIALIZATION_FAILED = 203
    | CURAND_ARCH_MISMATCH = 204
    | CURAND_INTERNAL_ERROR = 999

type CUDARandomRngType = 
    | CURAND_TEST = 0
    | CURAND_PSEUDO_DEFAULT = 100
    | CURAND_PSEUDO_XORWOW = 101
    | CURAND_QUASI_DEFAULT = 200
    | CURAND_QUASI_SOBOL32 = 201
    | CURAND_QUASI_SCRAMBLED_SOBOL32 = 202
    | CURAND_QUASI_SOBOL64 = 203
    | CURAND_QUASI_SCRAMBLED_SOBOL64 = 204

type CUDARandomOrdering = 
    | CURAND_PSEUDO_BEST = 100
    | CURAND_PSEUDO_DEFAULT = 101
    | CURAND_PSEUDO_SEEDED = 102
    | CURAND_QUASI_DEFAULT = 201

type CUDADirectionVectorSet =
    | CURAND_VECTORS_32_JOEKUO6 = 101
    | CURAND_DIRECTION_VECTORS_32_JOEKUO6 = 102
    | CURAND_VECTORS_64_JOEKUO6 = 103
    | CURAND_DIRECTION_VECTORS_64_JOEKUO6 = 104

[<Struct>]
type RandGenerator = 
    val handle : uint32

[<Struct>]
type RandDirectionVectors32 = 
    [<MarshalAs(UnmanagedType.ByValArray, SizeConst = 32)>]
    val direction_vectors :  uint32[]
[<Struct>]
type RandDirectionVectors64 = 
    [<MarshalAs(UnmanagedType.ByValArray, SizeConst = 64)>]
    val direction_vectors :  uint64[]

module CUDARandomDriver32 = 
    [<Literal>]
    let dllName =  "curand32_60"        
    //let dllName =  "curand32_40_17"        
    [<DllImport(dllName)>]
    extern curandStatus curandCreateGenerator(RandGenerator& generator, CUDARandomRngType rng_type);
    [<DllImport(dllName)>]
    extern curandStatus curandCreateGeneratorHost(RandGenerator& generator, CUDARandomRngType rng_type);
    [<DllImport(dllName)>]
    extern curandStatus curandDestroyGenerator(RandGenerator generator);
    [<DllImport(dllName)>]
    extern curandStatus curandGenerate(RandGenerator generator, IntPtr outputPtr, SizeT num);
    [<DllImport(dllName)>]
    extern curandStatus curandGenerateLogNormal(RandGenerator generator, IntPtr outputPtr, SizeT n, float mean, float stddev);
    [<DllImport(dllName)>]
    extern curandStatus curandGenerateLogNormalDouble(RandGenerator generator, IntPtr outputPtr, SizeT n, double mean, double stddev);
    [<DllImport(dllName)>]
    extern curandStatus curandGenerateLongLong(RandGenerator generator, IntPtr outputPtr, SizeT num);
    [<DllImport(dllName)>]
    extern curandStatus curandGenerateNormal(RandGenerator generator, IntPtr outputPtr, SizeT n, float mean, float stddev);
    [<DllImport(dllName)>]
    extern curandStatus curandGenerateNormalDouble(RandGenerator generator, IntPtr outputPtr, SizeT n, double mean, double stddev);
    [<DllImport(dllName)>]
    extern curandStatus curandGenerateSeeds(RandGenerator generator);
    [<DllImport(dllName)>]
    extern curandStatus curandGenerateUniform(RandGenerator generator, IntPtr outputPtr, SizeT num);
    [<DllImport(dllName)>]
    extern curandStatus curandGenerateUniformDouble(RandGenerator generator, IntPtr outputPtr, SizeT num);
    [<DllImport(dllName)>]
    extern curandStatus curandGetDirectionVectors32(RandDirectionVectors32& vectors, CUDADirectionVectorSet set);
    [<DllImport(dllName)>]
    extern curandStatus curandGetDirectionVectors64(RandDirectionVectors64& vectors, CUDADirectionVectorSet set);
    [<DllImport(dllName)>]
    extern curandStatus curandGetScrambleConstants32(IntPtr& constants);
    [<DllImport(dllName)>]
    extern curandStatus curandGetScrambleConstants64(IntPtr& constants);
    [<DllImport(dllName)>]
    extern curandStatus curandGetVersion(int& version);
    [<DllImport(dllName)>]
    extern curandStatus curandSetGeneratorOffset(RandGenerator generator, uint64 offset);
    [<DllImport(dllName)>]
    extern curandStatus curandSetGeneratorOrdering(RandGenerator generator, CUDARandomOrdering order);
    [<DllImport(dllName)>]
    extern curandStatus curandSetPseudoRandomGeneratorSeed(RandGenerator generator, uint64 seed);
    [<DllImport(dllName)>]
    extern curandStatus curandSetQuasiRandomGeneratorDimensions(RandGenerator generator, uint32 num_dimensions);
    [<DllImport(dllName)>]
    extern curandStatus curandSetStream(RandGenerator generator, CUDAStream stream);

    let CreateGenerator(rng_type) =
        let mutable generator = Unchecked.defaultof<RandGenerator>
        let r = curandCreateGenerator(&generator, rng_type)
        (r, generator)
    let DestroyGenerator(generator) =
        curandDestroyGenerator(generator)
    let SetPseudoRandomGeneratorSeed(generator, seed) = 
        curandSetPseudoRandomGeneratorSeed(generator, seed)    
    let SetGeneratorOffset(generator, offset) =
        curandSetGeneratorOffset(generator, offset)
    let SetGeneratorOrdering(generator, order) = 
        curandSetGeneratorOrdering(generator, order)
    let SetQuasiRandomGeneratorDimensions(generator, dimensions) = 
        curandSetQuasiRandomGeneratorDimensions(generator, dimensions)

    let CopyToHost(out:'T array, cudaPtr:CUDAPointer) = 
        let devPtr = cudaPtr.Pointer
        let outputPtr = GCHandle.Alloc(out, GCHandleType.Pinned).AddrOfPinnedObject()
        let unitSize = Marshal.SizeOf(typeof<float32>)
        let n = out.Length
        let size = SizeT(n * unitSize)
        let r = CUDARuntime32.cudaMemcpy(outputPtr, devPtr, size, CUDAMemcpyKind.cudaMemcpyDeviceToHost)
        r

    let GenerateUniform(generator, n:int) = 
        let unitSize = Marshal.SizeOf(typeof<float32>)
        let size = SizeT(n * unitSize)
        let mutable devicePtr = Unchecked.defaultof<IntPtr>
        let r = CUDARuntime32.cudaMalloc(&devicePtr, size)
        let r = curandGenerateUniform(generator, devicePtr, size)
        (r, CUDAPointer(devicePtr))
    let GenerateUniformDouble(generator, n:int) = 
        let unitSize = Marshal.SizeOf(typeof<float>)
        let size = SizeT(n * unitSize)
        let mutable devicePtr = Unchecked.defaultof<IntPtr>
        let r = CUDARuntime32.cudaMalloc(&devicePtr, size)
        let r = curandGenerateUniform(generator, devicePtr, size)
        (r, CUDAPointer(devicePtr))
    let GenerateNormal(generator, n:int, mean, stddev) = 
        let unitSize = Marshal.SizeOf(typeof<float32>)
        let size = SizeT(n * unitSize)
        let mutable devicePtr = Unchecked.defaultof<IntPtr>
        let r = CUDARuntime32.cudaMalloc(&devicePtr, size)
        let r = curandGenerateNormal(generator, devicePtr, size, mean, stddev)
        (r, CUDAPointer(devicePtr))
    let GenerateNormalDouble(generator, n:int, mean, stddev) = 
        let unitSize = Marshal.SizeOf(typeof<float32>)
        let size = SizeT(n * unitSize)
        let mutable devicePtr = Unchecked.defaultof<IntPtr>
        let r = CUDARuntime32.cudaMalloc(&devicePtr, size)
        let r = curandGenerateNormalDouble(generator, devicePtr, size, mean, stddev)
        (r, CUDAPointer(devicePtr))
    let GenerateLogNormal(generator, n:int, mean, stddev) = 
        let unitSize = Marshal.SizeOf(typeof<float32>)
        let size = SizeT(n * unitSize)
        let mutable devicePtr = Unchecked.defaultof<IntPtr>
        let r = CUDARuntime32.cudaMalloc(&devicePtr, size)
        let r = curandGenerateLogNormal(generator, devicePtr, size, mean, stddev)
        (r, CUDAPointer(devicePtr))
    let GenerateLogNormalDouble(generator, n:int, mean, stddev) = 
        let unitSize = Marshal.SizeOf(typeof<float>)
        let size = SizeT(n * unitSize)
        let mutable devicePtr = Unchecked.defaultof<IntPtr>
        let r = CUDARuntime32.cudaMalloc(&devicePtr, size)
        let r = curandGenerateLogNormalDouble(generator, devicePtr, size, mean, stddev)
        (r, CUDAPointer(devicePtr))

module CUDARandomDriver64 = 
    [<Literal>]
    let dllName =  "curand64_60"        
    [<DllImport(dllName)>]
    extern curandStatus curandCreateGenerator(RandGenerator& generator, CUDARandomRngType rng_type);
    [<DllImport(dllName)>]
    extern curandStatus curandCreateGeneratorHost(RandGenerator& generator, CUDARandomRngType rng_type);
    [<DllImport(dllName)>]
    extern curandStatus curandDestroyGenerator(RandGenerator generator);
    [<DllImport(dllName)>]
    extern curandStatus curandGenerate(RandGenerator generator, IntPtr outputPtr, SizeT num);
    [<DllImport(dllName)>]
    extern curandStatus curandGenerateLogNormal(RandGenerator generator, IntPtr outputPtr, SizeT n, float mean, float stddev);
    [<DllImport(dllName)>]
    extern curandStatus curandGenerateLogNormalDouble(RandGenerator generator, IntPtr outputPtr, SizeT n, double mean, double stddev);
    [<DllImport(dllName)>]
    extern curandStatus curandGenerateLongLong(RandGenerator generator, IntPtr outputPtr, SizeT num);
    [<DllImport(dllName)>]
    extern curandStatus curandGenerateNormal(RandGenerator generator, IntPtr outputPtr, SizeT n, float mean, float stddev);
    [<DllImport(dllName)>]
    extern curandStatus curandGenerateNormalDouble(RandGenerator generator, IntPtr outputPtr, SizeT n, double mean, double stddev);
    [<DllImport(dllName)>]
    extern curandStatus curandGenerateSeeds(RandGenerator generator);
    [<DllImport(dllName)>]
    extern curandStatus curandGenerateUniform(RandGenerator generator, IntPtr outputPtr, SizeT num);
    [<DllImport(dllName)>]
    extern curandStatus curandGenerateUniformDouble(RandGenerator generator, IntPtr outputPtr, SizeT num);
    [<DllImport(dllName)>]
    extern curandStatus curandGetDirectionVectors32(RandDirectionVectors32& vectors, CUDADirectionVectorSet set);
    [<DllImport(dllName)>]
    extern curandStatus curandGetDirectionVectors64(RandDirectionVectors64& vectors, CUDADirectionVectorSet set);
    [<DllImport(dllName)>]
    extern curandStatus curandGetScrambleConstants32(IntPtr& constants);
    [<DllImport(dllName)>]
    extern curandStatus curandGetScrambleConstants64(IntPtr& constants);
    [<DllImport(dllName)>]
    extern curandStatus curandGetVersion(int& version);
    [<DllImport(dllName)>]
    extern curandStatus curandSetGeneratorOffset(RandGenerator generator, uint64 offset);
    [<DllImport(dllName)>]
    extern curandStatus curandSetGeneratorOrdering(RandGenerator generator, CUDARandomOrdering order);
    [<DllImport(dllName)>]
    extern curandStatus curandSetPseudoRandomGeneratorSeed(RandGenerator generator, uint64 seed);
    [<DllImport(dllName)>]
    extern curandStatus curandSetQuasiRandomGeneratorDimensions(RandGenerator generator, uint32 num_dimensions);
    [<DllImport(dllName)>]
    extern curandStatus curandSetStream(RandGenerator generator, CUDAStream stream)

    let CreateGenerator(rng_type) =
        let mutable generator = Unchecked.defaultof<RandGenerator>
        let r = curandCreateGenerator(&generator, rng_type)
        (r, generator)
    let DestroyGenerator(generator) =
        curandDestroyGenerator(generator)
    let SetPseudoRandomGeneratorSeed(generator, seed) = 
        curandSetPseudoRandomGeneratorSeed(generator, seed)    
    let SetGeneratorOffset(generator, offset) =
        curandSetGeneratorOffset(generator, offset)
    let SetGeneratorOrdering(generator, order) = 
        curandSetGeneratorOrdering(generator, order)
    let SetQuasiRandomGeneratorDimensions(generator, dimensions) = 
        curandSetQuasiRandomGeneratorDimensions(generator, dimensions)

    let GenerateUniform(generator, n:int) = 
        let unitSize = Marshal.SizeOf(typeof<float32>)
        let size = SizeT(n * unitSize)
        let mutable devicePtr = Unchecked.defaultof<IntPtr>
        let r = CUDARuntime64.cudaMalloc(&devicePtr, size)
        let r = curandGenerateUniform(generator, devicePtr, size)
        (r, CUDAPointer(devicePtr))
    let GenerateUniformDouble(generator, n:int) = 
        let unitSize = Marshal.SizeOf(typeof<float>)
        let size = SizeT(n * unitSize)
        let mutable devicePtr = Unchecked.defaultof<IntPtr>
        let r = CUDARuntime64.cudaMalloc(&devicePtr, size)
        let r = curandGenerateUniform(generator, devicePtr, size)
        (r, CUDAPointer(devicePtr))
    let GenerateNormal(generator, n:int, mean, stddev) = 
        let unitSize = Marshal.SizeOf(typeof<float32>)
        let size = SizeT(n * unitSize)
        let mutable devicePtr = Unchecked.defaultof<IntPtr>
        let r = CUDARuntime64.cudaMalloc(&devicePtr, size)
        let r = curandGenerateNormal(generator, devicePtr, size, mean, stddev)
        (r, CUDAPointer(devicePtr))
    let GenerateNormalDouble(generator, n:int, mean, stddev) = 
        let unitSize = Marshal.SizeOf(typeof<float32>)
        let size = SizeT(n * unitSize)
        let mutable devicePtr = Unchecked.defaultof<IntPtr>
        let r = CUDARuntime64.cudaMalloc(&devicePtr, size)
        let r = curandGenerateNormalDouble(generator, devicePtr, size, mean, stddev)
        (r, CUDAPointer(devicePtr))
    let GenerateLogNormal(generator, n:int, mean, stddev) = 
        let unitSize = Marshal.SizeOf(typeof<float32>)
        let size = SizeT(n * unitSize)
        let mutable devicePtr = Unchecked.defaultof<IntPtr>
        let r = CUDARuntime64.cudaMalloc(&devicePtr, size)
        let r = curandGenerateLogNormal(generator, devicePtr, size, mean, stddev)
        (r, CUDAPointer(devicePtr))
    let GenerateLogNormalDouble(generator, n:int, mean, stddev) = 
        let unitSize = Marshal.SizeOf(typeof<float>)
        let size = SizeT(n * unitSize)
        let mutable devicePtr = Unchecked.defaultof<IntPtr>
        let r = CUDARuntime64.cudaMalloc(&devicePtr, size)
        let r = curandGenerateLogNormalDouble(generator, devicePtr, size, mean, stddev)
        (r, CUDAPointer(devicePtr))

/// the interface is present, but not decide how to use it
/// leave it here for now.
//type ICUDARandom = 
//    abstract member CreateGenerator : CUDARandomRngType -> curandStatus * RandGenerator
//    abstract member DestroyGenerator: RandGenerator -> curandStatus
//    abstract member SetPseudoRandomGeneratorSeed : RandGenerator * obj -> curandStatus
//    abstract member SetGeneratorOffset : RandGenerator * obj -> curandStatus
//    abstract member SetGeneratorOrdering : RandGenerator * CUDARandomOrdering -> curandStatus
//    abstract member SetQuasiRandomGeneratorDimensions : RandGenerator * obj -> curandStatus
//    abstract member GenerateUniform : RandGenerator * int -> curandStatus * CUDAPointer
//    abstract member GenerateUniformDouble : RandGenerator * int -> curandStatus * CUDAPointer
//    abstract member GenerateNormal : RandGenerator * int * double * double -> curandStatus * CUDAPointer
//    abstract member GenerateNormalDouble : RandGenerator * int  * double * double -> curandStatus * CUDAPointer
//    abstract member GenerateLogNormal : RandGenerator * int  * double * double -> curandStatus * CUDAPointer
//    abstract member GenerateLogNormalDouble : RandGenerator * int  * double * double -> curandStatus * CUDAPointer

type CUDARandom() = 
    let is64bit = IntPtr.Size = 8
    member this.CreateGenerator(rand_type) = 
        if is64bit then CUDARandomDriver64.CreateGenerator(rand_type)
        else CUDARandomDriver32.CreateGenerator(rand_type)
    member this.DestroyGenerator(g) = 
        if is64bit then CUDARandomDriver64.DestroyGenerator(g)
        else CUDARandomDriver32.DestroyGenerator(g)
    member this.SetPseudoRandomGeneratorSeed(g, obj) =
        if is64bit then CUDARandomDriver64.SetPseudoRandomGeneratorSeed(g, obj |> unbox |> uint64)
        else CUDARandomDriver32.SetPseudoRandomGeneratorSeed(g, obj |> unbox |> uint64)
    member this.SetGeneratorOffset(g, obj) = 
        if is64bit then CUDARandomDriver64.SetGeneratorOffset(g, obj |> unbox |> uint64)
        else CUDARandomDriver32.SetGeneratorOffset(g, obj |> unbox |> uint64)
    member this.SetGeneratorOrdering(g, ordering) =
        if is64bit then CUDARandomDriver64.SetGeneratorOrdering(g, ordering)
        else CUDARandomDriver32.SetGeneratorOrdering(g, ordering)
    member this.SetQuasiRandomGeneratorDimensions(g, obj) = 
        if is64bit then CUDARandomDriver64.SetQuasiRandomGeneratorDimensions(g, obj |> unbox |> uint32)
        else CUDARandomDriver32.SetQuasiRandomGeneratorDimensions(g, obj |> unbox |> uint32)
    member this.GenerateUniform(g, seed) = 
        if is64bit then CUDARandomDriver64.GenerateUniform(g, seed)
        else CUDARandomDriver32.GenerateUniform(g, seed)
    member this.GenerateUniformDouble(g, seed) = 
        if is64bit then CUDARandomDriver64.GenerateUniformDouble(g, seed)
        else CUDARandomDriver32.GenerateUniformDouble(g, seed)
    member this.GenerateNormal(g, seed, mean, variance) = 
        if is64bit then CUDARandomDriver64.GenerateNormal(g, seed, mean, variance)
        else CUDARandomDriver32.GenerateNormal(g, seed, mean, variance)
    member this.GenerateNormalDouble(g, seed, mean, variance) = 
        if is64bit then CUDARandomDriver64.GenerateNormalDouble(g, seed, mean, variance)
        else CUDARandomDriver32.GenerateNormalDouble(g, seed, mean, variance)
    member this.GenerateLogNormal(g, seed, mean, variance) = 
        if is64bit then CUDARandomDriver64.GenerateLogNormal(g, seed, mean, variance)
        else CUDARandomDriver32.GenerateLogNormal(g, seed, mean, variance)
    member this.GenerateLogNormalDouble(g, seed, mean, variance) = 
        if is64bit then CUDARandomDriver64.GenerateLogNormalDouble(g, seed, mean, variance)
        else CUDARandomDriver32.GenerateLogNormalDouble(g, seed, mean, variance)