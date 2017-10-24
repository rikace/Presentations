module Utils

    open System

    type Agent<'a> = MailboxProcessor<'a>

    type private ThreadSafeRandomRequest =
    | GetDouble of AsyncReplyChannel<decimal>
    let private threadSafeRandomAgent = Agent.Start(fun inbox -> 
            let rnd = new Random()
            let rec loop() = async {
                let! GetDouble(reply) = inbox.Receive() 
                reply.Reply((rnd.Next(-5, 5) |> decimal))
                return! loop()
            }
            loop() )


    let updatePrice (price:decimal) =
                let newPrice' = price + (threadSafeRandomAgent.PostAndReply(GetDouble))
                if newPrice' < 0m then 5m
                elif newPrice' > 50m then 45m
                else newPrice'



