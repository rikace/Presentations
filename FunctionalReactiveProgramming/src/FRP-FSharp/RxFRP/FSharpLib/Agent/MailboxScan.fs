namespace Easja360
//
//module MailboxScanOperation =
//
//    type Message =
//        | Message1
//        | Message2 of int
//        | Message3 of string
//
//    let agent =
//        MailboxProcessor.Start(fun inbox ->
//            let rec loop() =
//                inbox.Scan(function
//                    | Message1 ->
//                       Some (async { do printfn "message 1!"
//                                     return! loop() })
//                    | Message2 n ->
//                       Some (async { do printfn "message 2!"
//                                     return! loop() })
//                    | Message3 _ ->
//                       None)
//            loop())
//
//    agent.Post(Message1)
//    agent.Post(Message2(100))
//    agent.Post(Message3("abc"))
//    agent.Post(Message2(100))
//    let count = agent.CurrentQueueLength
//
