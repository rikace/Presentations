namespace Easj360FSharp.MailBoxProcess

module MailBoxProcess =

    let interactiveFunction maxI f f_zero =
        MailboxProcessor.Start(fun inbox ->
        let rec loop x i =
            async { 
                let! tryMsg =
                    if i < maxI then
                        async {
                            let! msg = inbox.TryReceive(timeout=0)
                            return msg
                        }
                    else
                        async {
                            let! msg = inbox.Receive()
                            return Some(msg)
                        }
                match tryMsg with
                | None -> return! loop (f x) (i+1)
                | Some msg -> ()
            }
        loop f_zero 0) 

    let interactiveFunction2 maxI f f_zero =
        MailboxProcessor.Start(fun inbox ->
        let rec loop x i =
            async { 
                let! tryMsg =
                    if i < maxI then
                        inbox.TryReceive(timeout=0)
                    else
                        async {
                            let! msg = inbox.Receive()
                            return Some(msg)
                        }
                match tryMsg with
                | None -> return! loop (f x) (i+1)
                | Some msg -> ()
            }
        loop f_zero 0)
