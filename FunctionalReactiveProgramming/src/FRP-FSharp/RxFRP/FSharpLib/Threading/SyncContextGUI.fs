
namespace Easj360FSharp
open System
open System.Threading
open System.ComponentModel


module SyncContextGUI = 
    
    let RaiseEventOnGuiThread (syncCtx:System.Threading.SynchronizationContext, event:Event<_>) args =
        try
            match syncCtx with 
            | null -> event.Trigger(args)
            | s -> s.Post(SendOrPostCallback(fun _ -> event.Trigger args),state=null)
        with
        |_ -> ()

    /////////////////////////////
    let context = AsyncOperationManager.SynchronizationContext
    let runInGuiContext f =
        context.Post(new SendOrPostCallback(fun _ -> f()), null)
        //context.Post(SendOrPostCallback(fun _ -> event.Trigger args),state=null)
                                                         
    /////////////////////////////
    let guiDispatch = Windows.Threading.Dispatcher.CurrentDispatcher
    let triggerGuiEvent (e:Event<_>) v = 
        guiDispatch.Invoke(new Action(fun () -> 
        e.Trigger(v) ), [| |]) |> ignore

    /////////////////////////////
    let syncContext = System.Threading.SynchronizationContext.Current

    // Check that we are being called from a GUI thread
    do match syncContext with 
        | null -> failwith "Failed to capture the synchronization context of the calling thread. The System.Threading.SynchronizationContext.Current of the calling thread is null"
        | _ -> ()



        





