namespace Easj360FSharp

open System
open System.Threading

module AgentScheduler =
//Agent alias for MailboxProcessor
    type Agent<'T> = MailboxProcessor<'T>
 
/// Two types of Schedule messages that can be sent
    type ScheduleMessage<'a> =
      | Schedule of ('a -> unit) * 'a * TimeSpan * TimeSpan * CancellationTokenSource AsyncReplyChannel
      | ScheduleOnce of ('a -> unit) * 'a * TimeSpan * CancellationTokenSource AsyncReplyChannel
 
/// An Agent based scheduler
    type SchedulerAgent<'a>()= 
 
        let scheduleOnce delay msg receiver (cts: CancellationTokenSource)=
            async { do! Async.Sleep(delay)
                    if (cts.IsCancellationRequested)
                        then cts.Dispose()
                    else msg |> receiver }
 
        let scheduleMany initialDelay  msg receiver delayBetween cts=
            let rec loop time (cts: CancellationTokenSource) =
                async { do! Async.Sleep(time)
                        if (cts.IsCancellationRequested)
                            then cts.Dispose()
                        else msg |> receiver
                        return! loop delayBetween cts}
            loop initialDelay cts
 
        let scheduler = Agent.Start(fun inbox ->
            let rec loop() = async {
                let! msg = inbox.Receive()
                let cs = new CancellationTokenSource()
                match msg with
                |   Schedule(receiver, msg:'a, initialDelay, delayBetween, replyChan) ->
                        Async.StartImmediate(scheduleMany
                                         (int initialDelay.TotalMilliseconds)
                                         msg receiver
                                         (int delayBetween.TotalMilliseconds)
                                         cs )
                        replyChan.Reply(cs)
                        return! loop()
                | ScheduleOnce(receiver, msg:'a, delay, replyChan) ->
                        Async.StartImmediate(scheduleOnce
                                         (int delay.TotalMilliseconds)
                                         msg receiver
                                         cs)
                        replyChan.Reply(cs)
                        return! loop()
            }
            loop())
 
          ///Schedules a message to be sent to the receiver after the initialDelay.
          ///  If delaybetween is specified then the message is sent reoccuringly at the delaybetween interval.
        member this.Schedule(receiver, msg, initialDelay, ?delayBetween) =
            let buildMessage replyChan =
              match delayBetween with
              | Some(x) -> Schedule(receiver,msg,initialDelay, x, replyChan)
              | _ -> ScheduleOnce(receiver,msg,initialDelay, replyChan)
            scheduler.PostAndReply (fun replyChan -> replyChan |> buildMessage)


    // let t = s.Schedule((fun x -> printfn "%d" x), 9, TimeSpan.FromSeconds(1.0), TimeSpan.FromSeconds(1.0));;
