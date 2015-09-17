namespace FSharpWcfServiceApplicationTemplate

open System
open System.Threading

type AnalyzeResult =
    inherit FSharpAsyncResult
    
    [<DefaultValue>] 
    val mutable m_value : int
    [<DefaultValue>] 
    val mutable m_result : int

    new (callback:AsyncCallback, state:obj) =
        { inherit FSharpAsyncResult(callback, state) }

    member x.Value 
        with get() = x.m_value
        and set(value) = x.m_value <- value
    
    member x.Result
        with get() = x.m_result
        and set(value) = x.m_result <- value


        
        

