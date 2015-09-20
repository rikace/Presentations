module AgentDisposable

open System
open System.Threading

////////////////////////// Disposable Agent /////////////////
type AgentDisposable<'T>(f:MailboxProcessor<'T> -> Async<unit>, ?cancelToken:System.Threading.CancellationTokenSource) =
    let cancelToken = defaultArg cancelToken (new System.Threading.CancellationTokenSource())
    let agent = MailboxProcessor.Start(f, cancelToken.Token)
    member x.Agent = agent
    interface IDisposable with
        member x.Dispose() = (agent :> IDisposable).Dispose()
                             cancelToken.Cancel()

