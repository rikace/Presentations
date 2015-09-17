namespace Easj360FSharp

module CopyFileModule =

    open System.Runtime.InteropServices

    type  private HANDLE = nativeint

    [<System.FlagsAttribute>]
    type CopyProgressCallbackReason =  
        | CallbackChunkedFinished = 0x00000000 
        | CallbackStreamSwitch = 0x00000001

    [<System.FlagsAttribute>]
    type CopyFileFlags =  
        | FileFailIfExists = 0x00000001
        | FileRestartable = 0x00000002
        | FileOpenSourceForWrite = 0x00000004
        | FileAllowDecryptedDestination = 0x00000008

    [<System.FlagsAttribute>]
    type CopyProgressResult =
        | ProgressContinue = 0
        | ProgressCancel = 1
        | ProgressStop = 2
        |ProgressQuiet = 3

    type CopyProgressRoutine = delegate of int64 * int64 * int64 * int64 * uint32 * CopyProgressCallbackReason * HANDLE * HANDLE * HANDLE -> int64

    type RaiseFileCopyProgressEvent = delegate of string * int64 * int64 -> unit

    [<DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Auto)>]
    [<MarshalAs(UnmanagedType.Bool)>]
    extern bool CopyFileEx(string ExistingFileName, string NewFileName, CopyProgressRoutine ProgressRoutine, HANDLE Data, bool* Cancel , CopyFileFlags CopyFlags);

    type CopyFile = class
        val private source: string
        val private destination: string
        val private m_event : DelegateEvent<RaiseFileCopyProgressEvent>        
        val mutable private  copiedSize :int64 
        val mutable private previousCopiedSize : int64 
        val mutable private cancel : bool
    
        new(source:string, destination:string) =
            { 
                source = source; destination = destination;
                m_event = new DelegateEvent<RaiseFileCopyProgressEvent>(); 
                copiedSize = 0L; previousCopiedSize = 0L; cancel = false
            }

        member private x.checkForCancel =  
                        if x.cancel = false then 
                            int64(CopyProgressResult.ProgressContinue)
                        else
                            int64(CopyProgressResult.ProgressCancel)

        member private x.CopyProgressRoutineHandler = new CopyProgressRoutine(fun totalFileSize totalBytesTransferred streamSize 
                                                                                  streamBytesTransferred streamNumber callbackReason 
                                                                                  sourceFile destinationFile data ->
                      match callbackReason with
                            | var1 when var1 = CopyProgressCallbackReason.CallbackChunkedFinished ->
                                x.copiedSize <- totalBytesTransferred - x.previousCopiedSize
                                x.m_event.Trigger( [| x.destination; box totalBytesTransferred; box totalFileSize |] )
                                x.checkForCancel
                            | _ -> x.checkForCancel
                       )
                                           
        [<CLIEventAttribute>]
        member x.ProgressSingleFileEvent = x.m_event.Publish       

        member x.Cancel 
            with get() = x.cancel
            and set(v) = x.cancel <- v
          
        member x.Start(fileFlags:CopyFileFlags, runAsync:bool) =
            match runAsync with
            | true -> Async.RunSynchronously( async { let result =CopyFileEx(x.source,x.destination, x.CopyProgressRoutineHandler, System.IntPtr.Zero, &&x.cancel, fileFlags)
                                                      return result } )
            | false -> CopyFileEx(x.source,x.destination, x.CopyProgressRoutineHandler, System.IntPtr.Zero, &&x.cancel, fileFlags) 
        end    

    type CopyDirectory = class
        val private source: string
        val private destination: string
        val private m_event : DelegateEvent<System.ComponentModel.ProgressChangedEventHandler>
        val mutable private  copied :int    
        val mutable private cancel : bool
    
        new(source:string, destination:string) =
            { 
                source = source; destination = destination;
                m_event = new DelegateEvent<System.ComponentModel.ProgressChangedEventHandler>()
                copied = 0; cancel = false
            }
                                            
        [<CLIEventAttribute>]
        member x.ProgressCopiedFileEvent = x.m_event.Publish    

        member x.Start() =
            let dirSource = new System.IO.DirectoryInfo(x.source)
            let dirDestination = new System.IO.DirectoryInfo(x.destination)
            dirSource.EnumerateFiles("*.*", System.IO.SearchOption.AllDirectories)
            |> Seq.map (fun s -> CopyFile(s.FullName, System.IO.Path.Combine(x.destination, s.Name)))
            |> Seq.map (fun s -> async { if s.Start(CopyFileFlags.FileFailIfExists, false) 
                                            then x.m_event.Trigger([|null; new System.ComponentModel.ProgressChangedEventArgs(System.Threading.Interlocked.Increment(&x.copied), null)|]) } )
            |> Async.Parallel
            |> Async.RunSynchronously
            |> ignore    
    end