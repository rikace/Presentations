#r "..\..\Lib\CUDA.NET.dll"
#r "..\..\Lib\FsGPU.Helpers.dll"
#r "FSharp.PowerPack.dll"
#nowarn "62"
open Microsoft.FSharp.Control
open System
open System.Threading
open GASS.CUDA
open GASS.CUDA.Engine
open GASS.CUDA.Types
open System.IO
open FsGPU.Cuda
open FsGPU.Helpers
open System.Collections
open System.Drawing
open System.Windows.Forms
open Microsoft.FSharp.Math
open System.Runtime.InteropServices

module BitonicSort =

    // Init CUDA, select 1st device.
    let cuda = new CUDA(0, true);

    // load module
    let m = cuda.LoadModule(Path.Combine(Environment.CurrentDirectory, "bitonic.cubin"))
    let func = cuda.GetModuleFunction("bitonicSort")

    let initKernel kernelFunc values = 
        let num : int = Array.length values
    
        // initialize kernel execution parameters
        // bitonicSort<<<1, dim3(blockDimX, blockDimY, 1), sizeof(int) * NUM>>>(dvalues);
        cuda.SetParameterSize(kernelFunc, uint32(sizeof<int>) + uint32(IntPtr.Size))
        cuda.SetFunctionBlockShape(kernelFunc, num, 1, 1)    
        cuda.SetFunctionSharedSize(kernelFunc, uint32(sizeof<int> * num))  // allocate shared memory which is needed by the kernel
        cuda.CopyHostToDevice<int>(values)


    let bitonic_sort n values = 
        let num = Array.length values
        let sortedValues = Array.create num 0
        let mutable dvalues = initKernel func values
        try
            cuda.SetParameter(func, 0, uint32(num))
            cuda.SetParameter(func, sizeof<int>, uint32(dvalues.Pointer))
        
            // GPU computation
            // launch kernel with the same data many times (sort array inplace)
            for i = 1 to n do
                cuda.Launch(func, 1, 1)
        
            cuda.CopyDeviceToHost<int>(dvalues, sortedValues);
            sortedValues
        finally
            cuda.Free(dvalues);

    let bitonic_sort_with_device_copy n values = 
        let num = Array.length values
        let sortedValues = Array.create num 0
        let mutable dvalues = initKernel func values
        let mutable dworkArr = cuda.CopyHostToDevice<int>(values)
        try
            cuda.SetParameter(func, 0, uint32(num))
            cuda.SetParameter(func, sizeof<int>, uint32(dworkArr.Pointer))
        
            // GPU computation
            for i = 1 to n do
                cuda.Launch(func, 1, 1)
                if i <> n then
                    cuda.CopyDeviceToDevice(dvalues, dworkArr, uint32(4 * num))

            cuda.CopyDeviceToHost<int>(dworkArr, sortedValues);
            sortedValues
        finally
            cuda.Free(dvalues)
            cuda.Free(dworkArr)

    let bitonic_sort_with_transfer_and_allocation n values = 
        let num = Array.length values
        let sortedValues = Array.create num 0
        let mutable dvalues = initKernel func values
        try
            for i = 1 to n do
                cuda.SetParameter(func, 0, uint32(num))
                cuda.SetParameter(func, sizeof<int>, uint32(dvalues.Pointer))           
                cuda.Launch(func, 1, 1)
                cuda.CopyDeviceToHost<int>(dvalues, sortedValues);
                if i <> n then
                    cuda.Free(dvalues) 
                // allocate memory and transfer original values for next iteration
                dvalues <- cuda.CopyHostToDevice<int>(values)           

            sortedValues
        finally
            cuda.Free(dvalues);

    let sequential_sort n (values : 't[]) = 
        let mutable res = null
        for i = 1 to n do
            res <- Array.copy values
            Array.Sort<'t>( res )
        res

    let checkSortOrder values =   
        let s = seq { for i = 1 to Array.length values - 1 do
                        if values.[i - 1] > values.[i] then
                            yield false 
                      yield true 
                    }
        Seq.head s

    let StartTesting =
    
        let NUM = 512   // NOTE: maximum number of CUDA threads in a block is 512
    
        // create values
        let rand = new Random();
        let values = Array.init NUM (fun i -> rand.Next())

        printfn "Running bitonic sort"
        let run repeats sortFunc title =
            printfn ""
            printfn title
            repeats |>
            List.iter (fun repeat ->
                printf "%d times: " repeat
                let sorted, span = MeasureTime (sortFunc repeat) values
                let gpuPassed = checkSortOrder sorted

                printfn "%d min, %d sec, %d ms; %s" span.Minutes span.Seconds span.Milliseconds (if gpuPassed then "PASSED" else "FAILED")
            )
    
        let repeats = [ 10; 100; 1000; 10000]
        run repeats bitonic_sort "GPU bitonic sort kernell"
        run repeats bitonic_sort_with_device_copy "GPU bitonic sort with device copy"
        run repeats bitonic_sort_with_transfer_and_allocation "GPU bitonic sort with transfer"
        run repeats sequential_sort "CPU sort"
    
