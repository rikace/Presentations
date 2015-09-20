namespace FSharpWcfServiceApplicationTemplate

open System.Runtime.Serialization
open System.ServiceModel
open System
open FSharpWcfServiceApplicationTemplate.Contracts

[<ServiceContract>]
type IService =
    [<OperationContract>]
    abstract GetData: value:int -> string
    [<OperationContract>]
    abstract GetDataUsingDataContract: value:int -> CompositeType
    [<OperationContract(AsyncPattern=true)>]
    abstract BeginAnalyze: value:int -> callback:AsyncCallback -> state:obj -> IAsyncResult
    abstract EndAnalyze: IAsyncResult -> int

type Service() =
    interface IService with
        member x.GetData value =
            sprintf "%A" value
        
        member x.GetDataUsingDataContract value =
            let composite = new CompositeType()
            composite.BoolValue <- false
            composite.StringValue <- "CIAO!!!!"
            composite

        member x.BeginAnalyze value callback state =
            let analyzeResult = new AnalyzeResult(callback, state)
            analyzeResult.Value <- value
            Threading.ThreadPool.QueueUserWorkItem((fun _ -> x.ExecuteTask(analyzeResult))) |> ignore 
            analyzeResult :> IAsyncResult
            
        member x.EndAnalyze iar =
            let mutable result = 0
            let analyzeResult = iar :?> AnalyzeResult
            try
                (analyzeResult :> IAsyncResult).AsyncWaitHandle.WaitOne()
                |> ignore
                result <- analyzeResult.Result               
                result
            finally
                analyzeResult.Dispose(false)

    member x.ExecuteTask(state:obj) =
            let analyzeResult = state :?> AnalyzeResult
            try                        
                let value = analyzeResult.Value
                System.Threading.Thread.Sleep(100)
                analyzeResult.Result <- value * value                
            finally
                analyzeResult.OnCompleted()

    member x.Squaree(v:int) =
            async {
                do! Async.Sleep(2000)
                let result = v * v
                return result
                }


//////////////////

open System.ServiceModel

[<ServiceContract>]
type IFSharpService =
    [<OperationContract>]
    abstract Multiplier : value:int -> string
   
type Message = int * AsyncReplyChannel<string>

[<SealedAttribute>]
[<ServiceBehaviorAttribute(ConcurrencyMode = ConcurrencyMode.Multiple, InstanceContextMode = InstanceContextMode.Single)>]
type FSharpService() =
    let agent = MailboxProcessor<Message>.Start(fun inbox ->
        let rec loop count =
            async {
                let! (msg, replay) = inbox.Receive()
                let res = (msg * msg) + count
                replay.Reply(res.ToString())
                return! loop(count + 1)          
            }
        loop 0)

    interface IFSharpService with
        member x.Multiplier(a) =
            let replay = agent.PostAndReply(fun replyChannel -> a, replyChannel)
            replay

