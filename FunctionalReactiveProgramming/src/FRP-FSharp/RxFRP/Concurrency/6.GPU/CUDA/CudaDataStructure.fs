namespace CudaDataStructure

open System
open System.Text
open System.Collections.Generic
open System.Runtime.InteropServices

type uint = uint32
type CUArrayFormat =
    | UNSIGNED_INT8 = 0x01
    | UNSIGNED_INT16 = 0x02
    | UNSIGNED_INT32 = 0x03
    | SIGNED_INT8 = 0x08
    | SIGNED_INT16 = 0x09
    | SIGNED_INT32 = 0x0a
    | HALF = 0x10
    | FLOAT = 0x20
type NumberOfChannelFormat = 
    | One = 1u
    | TWo = 2u
    | Four = 4u
type CUDAMemcpyKind = 
    | cudaMemcpyHostToHost = 0
    | cudaMemcpyHostToDevice = 1
    | cudaMemcpyDeviceToHost = 2
    | cudaMemcpyDeviceToDevice = 3

type GPUAttribute() = 
    inherit Attribute()

[<Struct>]
type CUDAModule = 
    val Pointer : IntPtr
[<Struct>]
type CUDAContext = 
    val Pointer : IntPtr
[<Struct>]
type CUDADevice = 
    val Pointer : int
[<Struct>]
type CUDAFunction = 
    val Pointer : IntPtr
[<Struct>]
type CUDAStream = 
    val Pointer : IntPtr
[<Struct>]
type CUDAPointer = 
    val Pointer : IntPtr    
    new(ptr) = { Pointer = ptr }
    new(cudaPointer:CUDAPointer) = { Pointer = cudaPointer.Pointer }
    member this.PointerSize with get() = IntPtr.Size

type CUDAPointer2<'T>(p:CUDAPointer) = 
    new(ptr:IntPtr) = CUDAPointer2(CUDAPointer(ptr))
    member this.Pointer with get() = p
    member this.PointerSize with get() = p.PointerSize
    member this.Is64Bit with get() = this.PointerSize = 8
    member this.Item
        with get (i:int) : float32 = failwith "for code generation only"
        and set (i:int) (v:float32) = failwith "for code generation only"
    member this.Set(x:float32, i:int) = failwith "for code generation only"

[<Struct>]
type ArrayDescriptor = 
     val Width : uint
     val Height : uint
     val Format : CUArrayFormat
     val NumChannels : NumberOfChannelFormat

type dim3 (x, y, z) = 
    new() = dim3(0,0,0)
    new(x) = dim3(x, 0, 0)
    new(x, y) = dim3(x, y, 0)
    member this.x with get() = x
    member this.y with get() = y
    member this.z with get() = z

type ThreadIdx() = 
    inherit dim3()
type BlockDim() =
    inherit dim3()

type CUResult = 
    | Success = 0
    | ErrorInvalidValue = 1
    | ErrorOutOfMemory = 2
    | ErrorNotInitialized = 3
    | ErrorDeinitialized = 4
    | ErrorNoDevice = 100
    | ErrorInvalidDevice = 101
    | ECCUncorrectable = 214
    | ErrorAlreadyAcquired = 210
    | ErrorAlreadyMapped = 208
    | ErrorArrayIsMapped = 207
    | ErrorContextAlreadyCurrent = 202    
    | ErrorFileNotFound = 301
    | ErrorInvalidImage = 200
    | ErrorInvalidContext = 201    
    | ErrorInvalidHandle = 400    
    | ErrorInvalidSource = 300    
    | ErrorLaunchFailed = 700
    | ErrorLaunchIncompatibleTexturing = 703
    | ErrorLaunchOutOfResources = 701
    | ErrorLaunchTimeout = 702
    | ErrorMapFailed = 205
    | ErrorNoBinaryForGPU = 209    
    | ErrorNotFound = 500    
    | ErrorNotMapped = 211
    | ErrorNotReady = 600        
    | ErrorUnmapFailed = 206
    | NotMappedAsArray = 212
    | NotMappedAsPointer = 213
    | PointerIs64Bit = 800
    | SizeIs64Bit = 801    
    | ErrorUnknown = 999

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
type SizeT= 
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
    member this.Name 
        with get() = String(this.nameChar).Trim()