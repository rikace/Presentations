#light


namespace FsGPU.Cuda

open Microsoft.FSharp.Control
open System
open System.Threading
open GASS.CUDA
open GASS.CUDA.Engine
open GASS.CUDA.Types
open System.IO
open System.Collections
open FsGPU

    
/// The internal type of messages for the agent
type CudaMsg<'args, 'res> = Run of 'args * AsyncReplyChannel<'res>  | Stop

type ExecutionContext = {
    mutable Execution : CUDAExecution; 
}

/// This class implements single threaded execution of CUDA kernels. 
/// According to "CUDA Programming Guide":
/// "... Several host threads can execute device code on the same device, but by design, a
/// host thread can execute device code on only one device. As a consequence, multiple
/// host threads are required to execute device code on multiple devices. Also, any
/// CUDA resources created through the runtime in one host thread cannot be used by
/// the runtime from another host thread. ..."

type CudaExecutor<'args, 'res>(deviceNum, cubinModulePath : string, funcName, 
                                executionFunc : (ExecutionContext*'args) -> 'res) as this =
    // instance fields
    let mutable _processor = None
    let mutable _execution = null
    
    // class fields
    static let _deviceCount = CudaHelpers.GetDeviceCount()
    static let _locks = new Object() |> Array.create(_deviceCount)
    static let _processors = None |> Array.create(_deviceCount)
    static let _cudas = None |> Array.create(_deviceCount)

    do this.start()

    member private this.start() = 
        lock _locks.[deviceNum] (fun () ->
            if _processors.[deviceNum] = None then
                _processors.[deviceNum] <- Some(MailboxProcessor.Start(fun inbox ->
                  // The single-threaded message-processing loop ...
                  let loop = seq {
                    while true do
                        let msg = Async.RunSynchronously(inbox.Receive())

                        let cuda = 
                            if _cudas.[deviceNum].IsNone then
                                // initialize device context
                                _cudas.[deviceNum] <- Some(new CUDA(deviceNum, true))
                            _cudas.[deviceNum].Value
                             
                        if _execution = null then
                            _execution <- new CUDAExecution( cuda, cubinModulePath, funcName)           
                        try 
                            match msg with
                            | Stop ->
                                // exit
                                match _cudas.[deviceNum] with | Some(cuda) -> cuda.Dispose() | _ -> ()
                                _processors.[deviceNum] <- None
                                yield () // break the loop
                            | Run(args,replyChannel) -> 
                                let res = executionFunc({ Execution = _execution; }, args)
                                do replyChannel.Reply(res)
                        finally
                            if _execution <> null then
                                _execution.Clear()    
                  }  
                  async {return (Seq.head loop |> ignore)} 
                ))
        )
        _processor <- _processors.[deviceNum]
    
    member this.GetInvoker() = 
        match _processor with  
        | Some(p) -> 
            fun args ->  p.PostAndReply(fun replyChannel -> Run(args, replyChannel)) 
        | _ -> failwith "not started"





