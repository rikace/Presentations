namespace CUDABLAS

//@@ need to decide what is the user scenario.

open System
open System.Text
open System.Collections.Generic
open System.Runtime.InteropServices
open CudaDataStructure

[<Struct>]
type CUDABLASHandle =
    val handle : uint32
[<Struct>]
type CUDAStream = 
    val Value : int
[<Struct>]
type CUDAFloatComplex =
    val real : float32
    val imag : float32
type CUBLASPointerMode = 
    | Host = 0
    | Device = 1
type CUBLASStatus = 
    | Success = 0
    | NotInitialized = 1
    | AllocFailed = 3
    | InvalidValue = 7
    | ArchMismatch = 8
    | MappingError = 11
    | ExecutionFailed = 13
    | InternalError = 14

module CUDABLASDriver64 = 
    [<Literal>]
    let dllName =  "cublas64_40_17" 
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasInit()
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasShutdown()
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasGetError()
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasFree(CUDAPointer devicePtr)
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasCreate(CUDABLASHandle& handle);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasDestroy(CUDABLASHandle handle);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasGetVersion(CUDABLASHandle handle, int& version);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasSetStream(CUDABLASHandle handle, CUDAStream streamId);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasGetStream(CUDABLASHandle handle, CUDAStream& streamId);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasGetPointerMode(CUDABLASHandle handle, CUBLASPointerMode& mode);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasSetPointerMode(CUDABLASHandle handle, CUBLASPointerMode mode);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasIcamax(CUDABLASHandle handle, int n, IntPtr x, int incx, int& result);
    [<DllImport(dllName)>]                                                                  
    extern CUBLASStatus cublasIdamax(CUDABLASHandle handle, int n, IntPtr x, int incx, int& result);
    [<DllImport(dllName)>]                                                                  
    extern CUBLASStatus cublasIsamax(CUDABLASHandle handle, int n, IntPtr x, int incx, int& result);
    [<DllImport(dllName)>]                                                                  
    extern CUBLASStatus cublasIzamax(CUDABLASHandle handle, int n, IntPtr x, int incx, int& result);
    [<DllImport(dllName)>]                                                                  
    extern CUBLASStatus cublasIcamin(CUDABLASHandle handle, int n, IntPtr x, int incx, int& result);
    [<DllImport(dllName)>]                                                                  
    extern CUBLASStatus cublasIdamin(CUDABLASHandle handle, int n, IntPtr x, int incx, int& result);
    [<DllImport(dllName)>]                                                                  
    extern CUBLASStatus cublasIsamin(CUDABLASHandle handle, int n, IntPtr x, int incx, int& result);
    [<DllImport(dllName)>]                                                                  
    extern CUBLASStatus cublasIzamin(CUDABLASHandle handle, int n, IntPtr x, int incx, int& result);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasSasum(CUDABLASHandle handle, int n, IntPtr x, int incx, float32& result);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasDasum(CUDABLASHandle handle, int n, IntPtr x, int incx, float& result);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasScasum(CUDABLASHandle handle, int n, IntPtr x, int incx, float32& result);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasDzasum(CUDABLASHandle handle, int n, IntPtr x, int incx, float& result);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasSaxpy(CUDABLASHandle handle, int n, float32& alpha, IntPtr x, int incx, IntPtr y, int incy);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasDaxpy(CUDABLASHandle handle, int n, float& alpha, IntPtr x, int incx, IntPtr y, int incy);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasCaxpy(CUDABLASHandle handle, int n, CUDAFloatComplex& alpha, IntPtr x, int incx, IntPtr y, int incy);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasZaxpy(CUDABLASHandle handle, int n, CUDAFloatComplex& alpha, IntPtr x, int incx, IntPtr y, int incy);    
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasScopy(CUDABLASHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasDcopy(CUDABLASHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasCcopy(CUDABLASHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasZcopy(CUDABLASHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasSdot(CUDABLASHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, float32& result);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasDdot(CUDABLASHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, float& result);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasCdotu(CUDABLASHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, CUDAFloatComplex& result);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasCdotc(CUDABLASHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, CUDAFloatComplex& result);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasZdotu(CUDABLASHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, CUDAFloatComplex& result);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasZdotc(CUDABLASHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, CUDAFloatComplex& result);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasSnrm2(CUDABLASHandle handle, int n, IntPtr x, int incx, float32&result);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasDnrm2(CUDABLASHandle handle, int n, IntPtr x, int incx, float& result);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasScnrm2(CUDABLASHandle handle, int n, IntPtr x, int incx, float32&result);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasDznrm2(CUDABLASHandle handle, int n, IntPtr x, int incx, float& result);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasSrot(CUDABLASHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, IntPtr c, IntPtr s);    
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasDrot(CUDABLASHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, IntPtr c, IntPtr s);    
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasCrot(CUDABLASHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, IntPtr c, IntPtr s);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasCsrot(CUDABLASHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, IntPtr c, IntPtr s);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasZrot(CUDABLASHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, IntPtr c, IntPtr s);    
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasZdrot(CUDABLASHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, IntPtr c, IntPtr s);    
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasSrotg(CUDABLASHandle handle, IntPtr a, IntPtr b, IntPtr c, IntPtr s);    
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasDrotg(CUDABLASHandle handle, IntPtr a, IntPtr b, IntPtr c, IntPtr s);    
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasCrotg(CUDABLASHandle handle, IntPtr a, IntPtr b, IntPtr c, IntPtr s);    
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasZrotg(CUDABLASHandle handle, IntPtr a, IntPtr b, IntPtr c, IntPtr s);    
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasSrotm(CUDABLASHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, IntPtr param);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasDrotm(CUDABLASHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, IntPtr param);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasSrotmg(CUDABLASHandle handle, IntPtr d1, IntPtr d2, IntPtr x1, IntPtr y1, IntPtr param);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasDrotmg(CUDABLASHandle handle, IntPtr d1, IntPtr d2, IntPtr x1, IntPtr y1, IntPtr param);    
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasSscal(CUDABLASHandle handle, int n, IntPtr alpha, IntPtr x, int incx);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasDscal(CUDABLASHandle handle, int n, IntPtr alpha, IntPtr x, int incx);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasCscal(CUDABLASHandle handle, int n, IntPtr alpha, IntPtr x, int incx);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasCsscal(CUDABLASHandle handle, int n, IntPtr alpha, IntPtr x, int incx);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasZscal(CUDABLASHandle handle, int n, IntPtr alpha, IntPtr x, int incx);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasZdscal(CUDABLASHandle handle, int n, IntPtr alpha, IntPtr x, int incx);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasSswap(CUDABLASHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasDswap(CUDABLASHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasCswap(CUDABLASHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasZswap(CUDABLASHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy);

module CUDABLASDriver64_2 = 
    [<Literal>]
    let dllName =  "cublas64_40_17" 
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasSrotmg(CUDABLASHandle handle, float32& d1, float32& d2, float32& x1, float32& y1, IntPtr param);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasDrotmg(CUDABLASHandle handle, float& d1, float& d2, float& x1, float& y1, IntPtr param);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasZrotg(CUDABLASHandle handle, CUDAFloatComplex& a, CUDAFloatComplex& b, float& c, CUDAFloatComplex& s);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasSrot(CUDABLASHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, float32&c, float32&s);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasCrotg(CUDABLASHandle handle, CUDAFloatComplex& a, CUDAFloatComplex& b, float32& c, CUDAFloatComplex& s);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasDrotg(CUDABLASHandle handle, float& a, float& b, float& c, float& s);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasSrotg(CUDABLASHandle handle, float32& a, float32& b, float32& c, float32& s);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasZdrot(CUDABLASHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, float& c, float& s);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasSaxpy(CUDABLASHandle handle, int n, IntPtr alpha, IntPtr x, int incx, IntPtr y, int incy);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasDaxpy(CUDABLASHandle handle, int n, IntPtr alpha, IntPtr x, int incx, IntPtr y, int incy);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasCaxpy(CUDABLASHandle handle, int n, IntPtr alpha, IntPtr x, int incx, IntPtr y, int incy);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasZaxpy(CUDABLASHandle handle, int n, IntPtr alpha, IntPtr x, int incx, IntPtr y, int incy);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasDrot(CUDABLASHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, float& c, float& s);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasCrot(CUDABLASHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, float32&c, CUDAFloatComplex& s);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasCsrot(CUDABLASHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, float32& c, float32& s);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasZrot(CUDABLASHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, float& c, CUDAFloatComplex& s);

module CUDABLASDriver32 = 
    [<Literal>]
    let dllName =  "cublas32_40_17" 
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasInit()
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasShutdown()
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasGetError()
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasFree( CUDAPointer devicePtr)
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasCreate(CUDABLASHandle& handle);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasDestroy(CUDABLASHandle handle);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasGetVersion(CUDABLASHandle handle, int& version);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasSetStream(CUDABLASHandle handle, CUDAStream streamId);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasGetStream(CUDABLASHandle handle, CUDAStream& streamId);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasGetPointerMode(CUDABLASHandle handle, CUBLASPointerMode& mode);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasSetPointerMode(CUDABLASHandle handle, CUBLASPointerMode mode);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasIcamax(CUDABLASHandle handle, int n, IntPtr x, int incx, int& result);
    [<DllImport(dllName)>]                                                                  
    extern CUBLASStatus cublasIdamax(CUDABLASHandle handle, int n, IntPtr x, int incx, int& result);
    [<DllImport(dllName)>]                                                                  
    extern CUBLASStatus cublasIsamax(CUDABLASHandle handle, int n, IntPtr x, int incx, int& result);
    [<DllImport(dllName)>]                                                                  
    extern CUBLASStatus cublasIzamax(CUDABLASHandle handle, int n, IntPtr x, int incx, int& result);
    [<DllImport(dllName)>]                                                                  
    extern CUBLASStatus cublasIcamin(CUDABLASHandle handle, int n, IntPtr x, int incx, int& result);
    [<DllImport(dllName)>]                                                                  
    extern CUBLASStatus cublasIdamin(CUDABLASHandle handle, int n, IntPtr x, int incx, int& result);
    [<DllImport(dllName)>]                                                                  
    extern CUBLASStatus cublasIsamin(CUDABLASHandle handle, int n, IntPtr x, int incx, int& result);
    [<DllImport(dllName)>]                                                                  
    extern CUBLASStatus cublasIzamin(CUDABLASHandle handle, int n, IntPtr x, int incx, int& result);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasSasum(CUDABLASHandle handle, int n, IntPtr x, int incx, float32& result);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasDasum(CUDABLASHandle handle, int n, IntPtr x, int incx, float& result);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasScasum(CUDABLASHandle handle, int n, IntPtr x, int incx, float32& result);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasDzasum(CUDABLASHandle handle, int n, IntPtr x, int incx, float& result);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasSaxpy(CUDABLASHandle handle, int n, float32& alpha, IntPtr x, int incx, IntPtr y, int incy);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasDaxpy(CUDABLASHandle handle, int n, float& alpha, IntPtr x, int incx, IntPtr y, int incy);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasCaxpy(CUDABLASHandle handle, int n, CUDAFloatComplex& alpha, IntPtr x, int incx, IntPtr y, int incy);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasZaxpy(CUDABLASHandle handle, int n, CUDAFloatComplex& alpha, IntPtr x, int incx, IntPtr y, int incy);    
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasScopy(CUDABLASHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasDcopy(CUDABLASHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasCcopy(CUDABLASHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasZcopy(CUDABLASHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasSdot(CUDABLASHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, float32& result);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasDdot(CUDABLASHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, float& result);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasCdotu(CUDABLASHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, CUDAFloatComplex& result);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasCdotc(CUDABLASHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, CUDAFloatComplex& result);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasZdotu(CUDABLASHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, CUDAFloatComplex& result);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasZdotc(CUDABLASHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, CUDAFloatComplex& result);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasSnrm2(CUDABLASHandle handle, int n, IntPtr x, int incx, float32&result);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasDnrm2(CUDABLASHandle handle, int n, IntPtr x, int incx, float& result);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasScnrm2(CUDABLASHandle handle, int n, IntPtr x, int incx, float32&result);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasDznrm2(CUDABLASHandle handle, int n, IntPtr x, int incx, float& result);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasSrot(CUDABLASHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, IntPtr c, IntPtr s);    
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasDrot(CUDABLASHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, IntPtr c, IntPtr s);    
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasCrot(CUDABLASHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, IntPtr c, IntPtr s);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasCsrot(CUDABLASHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, IntPtr c, IntPtr s);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasZrot(CUDABLASHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, IntPtr c, IntPtr s);    
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasZdrot(CUDABLASHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, IntPtr c, IntPtr s);    
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasSrotg(CUDABLASHandle handle, IntPtr a, IntPtr b, IntPtr c, IntPtr s);    
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasDrotg(CUDABLASHandle handle, IntPtr a, IntPtr b, IntPtr c, IntPtr s);    
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasCrotg(CUDABLASHandle handle, IntPtr a, IntPtr b, IntPtr c, IntPtr s);    
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasZrotg(CUDABLASHandle handle, IntPtr a, IntPtr b, IntPtr c, IntPtr s);    
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasSrotm(CUDABLASHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, IntPtr param);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasDrotm(CUDABLASHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, IntPtr param);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasSrotmg(CUDABLASHandle handle, IntPtr d1, IntPtr d2, IntPtr x1, IntPtr y1, IntPtr param);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasDrotmg(CUDABLASHandle handle, IntPtr d1, IntPtr d2, IntPtr x1, IntPtr y1, IntPtr param);    
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasSscal(CUDABLASHandle handle, int n, IntPtr alpha, IntPtr x, int incx);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasDscal(CUDABLASHandle handle, int n, IntPtr alpha, IntPtr x, int incx);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasCscal(CUDABLASHandle handle, int n, IntPtr alpha, IntPtr x, int incx);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasCsscal(CUDABLASHandle handle, int n, IntPtr alpha, IntPtr x, int incx);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasZscal(CUDABLASHandle handle, int n, IntPtr alpha, IntPtr x, int incx);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasZdscal(CUDABLASHandle handle, int n, IntPtr alpha, IntPtr x, int incx);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasSswap(CUDABLASHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasDswap(CUDABLASHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasCswap(CUDABLASHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasZswap(CUDABLASHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy);

module CUDABLASDriver32_2 = 
    [<Literal>]
    let dllName =  "cublas32_40_17" 
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasSrotmg(CUDABLASHandle handle, float32& d1, float32& d2, float32& x1, float32& y1, IntPtr param);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasDrotmg(CUDABLASHandle handle, float& d1, float& d2, float& x1, float& y1, IntPtr param);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasZrotg(CUDABLASHandle handle, CUDAFloatComplex& a, CUDAFloatComplex& b, float& c, CUDAFloatComplex& s);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasSrot(CUDABLASHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, float32&c, float32&s);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasCrotg(CUDABLASHandle handle, CUDAFloatComplex& a, CUDAFloatComplex& b, float32& c, CUDAFloatComplex& s);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasDrotg(CUDABLASHandle handle, float& a, float& b, float& c, float& s);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasSrotg(CUDABLASHandle handle, float32& a, float32& b, float32& c, float32& s);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasZdrot(CUDABLASHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, float& c, float& s);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasSaxpy(CUDABLASHandle handle, int n, IntPtr alpha, IntPtr x, int incx, IntPtr y, int incy);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasDaxpy(CUDABLASHandle handle, int n, IntPtr alpha, IntPtr x, int incx, IntPtr y, int incy);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasCaxpy(CUDABLASHandle handle, int n, IntPtr alpha, IntPtr x, int incx, IntPtr y, int incy);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasZaxpy(CUDABLASHandle handle, int n, IntPtr alpha, IntPtr x, int incx, IntPtr y, int incy);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasDrot(CUDABLASHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, float& c, float& s);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasCrot(CUDABLASHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, float32&c, CUDAFloatComplex& s);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasCsrot(CUDABLASHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, float32& c, float32& s);
    [<DllImport(dllName)>]
    extern CUBLASStatus cublasZrot(CUDABLASHandle handle, int n, IntPtr x, int incx, IntPtr y, int incy, float& c, CUDAFloatComplex& s);
