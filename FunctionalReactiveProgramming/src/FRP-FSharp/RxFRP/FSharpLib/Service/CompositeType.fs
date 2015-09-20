namespace FSharpWcfServiceApplicationTemplate.Contracts

open System.Runtime.Serialization
open System.ServiceModel

[<DataContract>]
type CompositeType() =
    [<DefaultValue(false)>] val mutable _boolValue : bool 
    [<DefaultValue(false)>] val mutable _stringValue : string 
    [<DataMember>] 
    member x.BoolValue with get() = x._boolValue and set(value) = x._boolValue <- value
    [<DataMember>] 
    member x.StringValue with get() = x._stringValue and set(value) = x._stringValue <- value

