open System

type msg1 =
    | Start
    | Exit
    | Reset
    | GoIdle
    | Next
    | Show of AsyncReplyChannel
    | State of AsyncReplyChannel


type msg2 =
    Two of int*AsyncReplyChannel | Three of int*AsyncReplyChannel
    


let agentCalc ()=
    MailboxProcessor.Start(fun i ->
        let rec running =
            async {
                let! msg= i.Receive()
                match msg with
                | Two (n,replyChannel) ->
                    do replyChannel.Reply(n*2)
                    return! running
                | Three (n, replyChannel) ->
                    do replyChannel.Reply(n*3)
                    return! running
                }
        running )

let agent2= agentCalc ()
let agent3= agentCalc ()

let mutable total=0
let agent1=
        MailboxProcessor.Start(fun i ->
            let rec idle (total:int)=
                async {
                    let! msg= i.Receive()
                    match msg with
                    | Reset ->
                        let total=0
                        return! idle total
                    | Exit ->
                        return ()
                    | Start ->
                        return! running total
                    | Show (replyChannel)->
                        do replyChannel.Reply(total)
                        return! idle total
                    | State (replyChannel)->
                        do replyChannel.Reply("idle")
                        return! idle total
                    | _->
                        return! idle total                        
                    }
            and running total= 
                async{
                    let! msg= i.Receive()
                    match msg with
                    | Next ->
                        let z=new Random()
                        let n1=z.Next(1)
                        let n2=z.Next(9)
                        let total=
                                if n1=0 then
                                    let r1=agent2.PostAndReply(fun replyChannel->Two(n2,replyChannel))
                                    let r2=agent3.PostAndReply(fun replyChannel->Two(n2,replyChannel))
                                    total+r1+r2
                                else
                                    let r1=agent2.PostAndReply(fun replyChannel->Three(n2,replyChannel))
                                    let r2=agent3.PostAndReply(fun replyChannel->Three(n2,replyChannel))
                                    total+r1+r2
                        return! running total
                    | GoIdle ->
                        return! idle total
                    | Show (replyChannel) ->
                        do replyChannel.Reply(total)
                        return! running total
                    | State (replyChannel)->
                        do replyChannel.Reply("running")
                        return! running total
                    | Reset ->
                        let total=0
                        return! running total
                    | _->
                        return! idle total                        
                }
            idle total)

			
			//////////////////////////

let r1 = agent1.PostAndReply(fun reply -> State(reply))
printfn "State %s" r1 |>ignore;;
State idle 

let r2 = agent1.PostAndReply(fun reply -> Show(reply))
printfn "Total %d" r2 |>ignore;;
Total 0 

agent1.Post(Start)
let r3 = agent1.PostAndReply(fun reply -> State(reply))
printfn "State %s" r3 |>ignore;;
State running 

for i in 1..5 do
    agent1.Post(Next)
    let r4 = agent1.PostAndReply(fun reply -> Show(reply))
    printfn "Total %d" r4 |>ignore;;

Total 24
Total 48
Total 72
Total 96
Total 120

agent1.Post(Reset)
let r5 = agent1.PostAndReply(fun reply -> Show(reply))
printfn "Total %d" r5 |>ignore;;
Total 0 

agent1.Post(GoIdle)
let r6 = agent1.PostAndReply(fun reply -> State(reply))
printfn "State %s" r6 |>ignore;;
State idle 

agent1.Post(Exit);;

