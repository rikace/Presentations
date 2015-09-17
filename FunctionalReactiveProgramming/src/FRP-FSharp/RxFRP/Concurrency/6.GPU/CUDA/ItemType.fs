module ItemType

open System
open System.Diagnostics
open System.Collections
open System.Collections.Generic
open System.IO
open Microsoft.FSharp.Quotations
open Microsoft.FSharp.Quotations.Patterns
open Microsoft.FSharp.Quotations.DerivedPatterns
open Microsoft.FSharp.Quotations.ExprShape

type Item(expr:Expr) = 
    let exp = expr

type Code() = 
    inherit List<string>()
    let mutable functionID = 0
    let mutable variableID = 0
    member this.FunctionID 
        with get () = functionID
        and set (v) = functionID <-v
    member this.VariableID 
        with get () = variableID
        and set(v) = variableID <- v
    member this.IncreaseFunctionID() = 
        functionID <- this.FunctionID + 1
    member this.IncreaseVariableID() = 
        variableID <- this.VariableID + 1

    member this.ToCode() = 
        "#include \"CUDALibrary.h\"\r\n" + String.Join("\r\n", this) + "\r\n\r\n\r\n"