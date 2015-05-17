module GPUTranslator

open System
open System.Reflection
open System.Collections
open System.Collections.Generic
open Microsoft.FSharp.Reflection
open Microsoft.FSharp.Quotations
open Microsoft.FSharp.Core
open Microsoft.FSharp.Quotations.Patterns
open Microsoft.FSharp.Quotations.DerivedPatterns
open Microsoft.FSharp.Quotations.ExprShape
open CudaDataStructure

let floatType = "double"

type Type with
    member this.HasInterface(t:Type) = 
        this.GetInterface(t.FullName) <> null

let rec translateFromNETType (t:Type) (a:string)= 
    if t = typeof<int> then "int"
    elif t = typeof<float32> then "float"
    elif t = typeof<float> then "double"
    elif t = typeof<bool> then "bool"
    elif t.IsArray then 
        let elementTypeString = translateFromNETType (t.GetElementType()) a
        sprintf "List<%s>" elementTypeString
    elif t.HasInterface(typeof<IEnumerable>) then 
        let elementTypeString = translateFromNETType (t.GetGenericArguments().[0]) a
        sprintf "List<%s>" elementTypeString
    elif t = typeof< Microsoft.FSharp.Core.unit > then String.Empty
    elif t = typeof< CUDAPointer2<float> > then sprintf "%s*" "double"
    elif t = typeof< CUDAPointer2<float32>> then sprintf "%s*" "float"
    elif t.Name = "FSharpFunc`2" then 
        let input = translateFromNETType (t.GetGenericArguments().[0]) a 
        let out = translateFromNETType (t.GetGenericArguments().[1]) a
        sprintf "%s(*%s)(%s)" input a out
    elif t = typeof<System.Void> then
        String.Empty
    else failwith "not supported type"

let translateFromNETTypeToFunReturn (t:Type) (a:string) = 
    let r = translateFromNETType t a
    if String.IsNullOrEmpty(r) then "void"
    else r

let translateFromNETTypeLength (t:Type) c = 
    if t.IsArray then
        sprintf ", int %A_len" c
    else
        String.Empty

let isValueType (t:Type) = 
    if t.IsValueType then true
    elif t.HasInterface(typeof<IEnumerable>) then false
    else failwith "is value type failed"