namespace FSharpWcfAgentService.Contracts

open System
open FSharpWcfServiceApplicationTemplate.Contracts
open System.ServiceModel

type AgentService() =
    let innerAgent =
            MailboxProcessor<IAgentCallBack * int>.Start(fun inbox ->
                let rec loop n =
                    async { let! msg = inbox.Receive()
                            let callBack = fst msg
                            let value = snd msg
                            let res = value * value
                            do! Async.Sleep(value)
                            callBack.GetDataCallBack(res.ToString())
                            return! loop(n + n)}
//                            match msg with
//                            | Die -> return ()
//                            | GetCounter(reply) ->  reply.Reply(n)
//                                                    return! loop(n + 1)
//                            | Incr x -> return! loop(n + x)
//                            | Fetch(x, reply) ->
//                                let res = x * x
//                                reply.Reply(res)
//                                return! loop(n + x)}
                loop 0)
    
    interface IAgent with
        
        member x.GetDataOneWatStr id name =
            "some value returned"

        member x.GetDataOneWay value =
            //System.Threading.Thread.Sleep(value)
            let callBack = OperationContext.Current.GetCallbackChannel<IAgentCallBack>()
            innerAgent.Post(callBack, value)
            //let res = value * value
            //x.CallBack.GetDataCallBack(res.ToString())

//    member private x.CallBack:IAgentCallBack =
//        OperationContext.Current.GetCallbackChannel<IAgentCallBack>()



            



