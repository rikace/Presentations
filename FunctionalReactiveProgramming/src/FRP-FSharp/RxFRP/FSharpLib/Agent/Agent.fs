namespace Easj360FSharp

module AgentHelper = 
    /// Type alias for F# mailbox processor type
    type Agent<'T> = MailboxProcessor<'T>


    let (<--) (a:Agent<_>) x = a.Post x

    let (<!-) (a:Agent<_>) x = a.PostAndReply x

