namespace AgentModule

[<AutoOpenAttribute>]
module Utils =

    open System

    let internal synchronize f = 
        let ctx = System.Threading.SynchronizationContext.Current 
        f (fun g ->
          let nctx = System.Threading.SynchronizationContext.Current 
          if ctx <> null && ctx <> nctx then ctx.Post((fun _ -> g()), null)
          else g() )

    
    type Agent<'T> = MailboxProcessor<'T>

    let (<--) (m:Agent<_>) msg = m.Post msg
    let (<->) (m:Agent<_>) msg = m.PostAndReply(fun replyChannel -> msg replyChannel)
    let (<-!) (m: Agent<_>) msg = m.PostAndAsyncReply(fun replyChannel -> msg replyChannel)