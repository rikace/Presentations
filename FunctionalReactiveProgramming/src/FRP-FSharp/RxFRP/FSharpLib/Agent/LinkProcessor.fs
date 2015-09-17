namespace Easj360FSharp

open System
open System.Diagnostics
open System.Threading

module LinkProcessor =
    

    type Message =  Phrase of string 
                    | Stop

    type ILinkProcessor =
        abstract Send : string -> Unit
        abstract Stop : Unit -> Unit

    let should_send_message_to_link_processor_if_it_contains_link () = 
        let latch = new AutoResetEvent(false)     
        let interceptedMessage = ref ""
     
        let linkProcessorStub = { 
            new ILinkProcessor  with
                member x.Send (message) =
                     interceptedMessage := message 
                     latch.Set() |> ignore
                member x.Stop() = ()  }
        
        let messageWithLink =""

        let wasTripped = latch.WaitOne(1000)
                         |> ignore
        ()
//        Assert.True(wasTripped)
//        Assert.Equal(messageWithLink.Text, !interceptedMessage)


    type LinkProcessor(callBack) =      
      let agent = MailboxProcessor.Start(fun inbox ->
        let rec loop () =
          async {
                  let! msg = inbox.Receive()
                  match msg with
                  | Phrase item ->
                    callBack item
                    return! loop()
                  | Stop ->
                    return ()
                }
        loop()
      ) 
      interface ILinkProcessor with 
            member x.Send(status:string) = agent.Post(Phrase(status))       
            member x.Stop() = agent.Post(Stop)
    
    type MainProcessor(linkProcessor:ILinkProcessor) =
      let agent = MailboxProcessor.Start(fun inbox ->
        let rec loop () =
          async {
                  let! msg = inbox.Receive()
                  match msg with
                  | Phrase item -> linkProcessor.Send(item)
                                   return! loop()                  
                  | Stop ->
                    return ()
                }
        loop()
      )
       
       member x.Send(statuses:seq<string>) =  statuses |> Seq.iter (fun status -> agent.Post(Phrase(status)))       
       member x.Stop() = agent.Post(Stop)
