namespace FSharpWcfAgentService.Contracts

open System.Runtime.Serialization
open System.ServiceModel
open System

type IAgentCallBack =
    [<OperationContract(IsOneWay=true)>]
    abstract GetDataCallBack: value:string -> unit


[<ServiceContract(CallbackContract=typeof<IAgentCallBack>)>]
type IAgent =
    [<OperationContract(IsOneWay=true)>]
    abstract GetDataOneWay: value:int -> unit

    [<OperationContract>]
    abstract GetDataOneWatStr : id : int -> name : string -> string




