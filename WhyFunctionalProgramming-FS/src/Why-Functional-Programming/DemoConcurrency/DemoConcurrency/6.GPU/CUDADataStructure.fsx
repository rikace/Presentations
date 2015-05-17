#I "C:\GPU\CUDA\bin"

module CUDADataStructure =
    open System
    open System.Runtime.InteropServices

    // CUDA error enumeration
    type cudaError =
        | cudaErrorAddressOfConstant = 22
        | cudaErrorApiFailureBase = 10000
        | cudaErrorCudartUnloading = 29
        | cudaErrorInitializationError = 3
        | cudaErrorInsufficientDriver = 35
        | cudaErrorInvalidChannelDescriptor = 20
        | cudaErrorInvalidConfiguration = 9
        | cudaErrorInvalidDevice = 10
        | cudaErrorInvalidDeviceFunction = 8
        | cudaErrorInvalidDevicePointer = 17
        | cudaErrorInvalidFilterSetting = 26
        | cudaErrorInvalidHostPointer = 16
        | cudaErrorInvalidMemcpyDirection = 21
        | cudaErrorInvalidNormSetting = 27
        | cudaErrorInvalidPitchValue = 12
        | cudaErrorInvalidResourceHandle = 33
        | cudaErrorInvalidSymbol = 13
        | cudaErrorInvalidTexture = 18
        | cudaErrorInvalidTextureBinding = 19
        | cudaErrorInvalidValue = 11
        | cudaErrorLaunchFailure = 4
        | cudaErrorLaunchOutOfResources = 7
        | cudaErrorLaunchTimeout = 6
        | cudaErrorMapBufferObjectFailed = 14
        | cudaErrorMemoryAllocation = 2
        | cudaErrorMemoryValueTooLarge = 32
        | cudaErrorMissingConfiguration = 1
        | cudaErrorMixedDeviceExecution = 28
        | cudaErrorNoDevice = 37
        | cudaErrorNotReady = 34
        | cudaErrorNotYetImplemented = 31
        | cudaErrorPriorLaunchFailure = 5
        | cudaErrorSetOnActiveProcess = 36
        | cudaErrorStartupFailure = 127
        | cudaErrorSynchronizationError = 25
        | cudaErrorTextureFetchFailed = 23
        | cudaErrorTextureNotBound = 24
        | cudaErrorUnknown = 30
        | cudaErrorUnmapBufferObjectFailed = 15
        | cudaErrorIncompatibleDriverContext = 49
        | cudaSuccess = 0
    [<Struct>]
    type SizeT =
        val value : IntPtr
        new (n:int) = { value = IntPtr(n) }
        new (n:int64) = { value = IntPtr(n) }

    [<Struct>]
    type CUDADeviceProp =
        [<MarshalAs(UnmanagedType.ByValArray, SizeConst = 256)>]
        val nameChar : char array
        val totalGlobalMem : SizeT
        val sharedMemPerBlock : SizeT
        val regsPerBlock : int
        val warpSize : int
        val memPitch : SizeT
        val maxThreadsPerBlock : int
        [<MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)>]
        val maxThreadsDim : int array
        [<MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)>]
        val maxGridSize : int array
        val clockRate : int
        val totalConstMem : SizeT
        val major : int
        val minor : int
        val textureAlignment: SizeT
        val deviceOverlap : int
        val multiProcessorCount : int
        val kernelExecTimeoutEnabled : int
        val integrated : int
        val canMapHostMemory : int
        val computeMode : int
        val maxTexture1D : int
        [<MarshalAs(UnmanagedType.ByValArray, SizeConst = 2)>]
        val maxTexture2D : int array
        [<MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)>]
        val maxTexture3D : int array
        [<MarshalAs(UnmanagedType.ByValArray, SizeConst = 2)>]
        val maxTexture1DLayered : int array
        [<MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)>]
        val maxTexture2DLayered : int array
        val surfaceAlignment : SizeT
        val concurrentKernels : int
        val ECCEnabled : int
        val pciBusID : int
        val pciDeviceID : int
        val pciDomainID : int
        val tccDriver : int
        val asyncEngineCount : int
        val unifiedAddressing : int
        val memoryClockRate : int
        val memoryBusWidth : int
        val l2CacheSize : int
        val maxThreadsPerMultiProcessor : int
        member this.Name = String(this.nameChar).Trim('\000')

    type cudaLimit =
        | cudaLimitStackSize = 0
        | cudaLimitPrintfFifoSize = 1
        | cudaLimitMallocHeapSize = 2

module CUDA32 =
    open System
    open System.Runtime.InteropServices
    open CUDADataStructure

    [<Literal>]
    let dllName = "cudart32_60"

    [<DllImport(dllName)>]
    extern cudaError cudaGetDeviceProperties(CUDADeviceProp& prop, int device)

    [<DllImport(dllName)>]
    extern cudaError cudaDeviceGetLimit(SizeT& pSize, cudaLimit limit)

module CUDA64 =
    open System
    open System.Runtime.InteropServices
    open CUDADataStructure

    [<Literal>]
    let dllName = "cudart64_60"

    [<DllImport(dllName)>]
    extern cudaError cudaGetDeviceProperties(CUDADeviceProp& prop, int device)

    [<DllImport(dllName)>]
    extern cudaError cudaDeviceGetLimit(SizeT& pSize, cudaLimit limit)


let main =
    let mutable prop = CUDADataStructure.CUDADeviceProp()

    // get the first graphic card by passing in 0
    let returnCode = CUDA32.cudaGetDeviceProperties (&prop, 0)
    let returnCode64 = CUDA64.cudaGetDeviceProperties (&prop, 0)

    printfn "%A - %A" returnCode prop.Name

    ignore <| System.Console.ReadKey()
    0 // return an integer exit code”
