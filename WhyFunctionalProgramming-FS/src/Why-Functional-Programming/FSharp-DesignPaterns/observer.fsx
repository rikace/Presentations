type Subject() = 
    let mutable notify = fun _ -> ()
    member this.Subscribe (notifyFunction) = 
        let wrap f i = f(i); i
        notify <- (wrap notifyFunction) >> notify
    member this.Reset() = notify <- fun _ -> ()
    member this.SomethingHappen(k) = 
        notify k

type ObserverA() =
    member this.NotifyMe(i) = printfn "notified A %A" i
type ObserverB() = 
    member this.NotifyMeB(i) = printfn "notified B %A" i

let observer() = 
    let a = ObserverA()
    let b = ObserverB()
    let subject = Subject()
    subject.Subscribe(a.NotifyMe)
    subject.Subscribe(b.NotifyMeB)
    subject.SomethingHappen("good")

observer()