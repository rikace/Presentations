module Observer

// define a subject
type Subject() = 
    // define a default notify function
    let mutable notify = fun _ -> ()

    // subscribe to a notification function
    member this.Subscribe notifyFunction = 
        let wrap f i = f i; i
        notify <- wrap notifyFunction >> notify

    // reset notification function
    member this.Reset() = notify <- fun _ -> ()

    // notify when something happens
    member this.SomethingHappen k = 
        notify k

// define observer A
type ObserverA() =
    member this.NotifyMe i = printfn "notified A %A" i

// define observer B
type ObserverB() = 
    member this.NotifyMeB i = printfn "notified B %A" i

// observer pattern
let observer() = 
    // create two observers
    let a = ObserverA()
    let b = ObserverB()

    // create a subject
    let subject = Subject()

    // let observer subscribe to subject
    subject.Subscribe a.NotifyMe
    subject.Subscribe b.NotifyMeB

    // something happens to the subject
    subject.SomethingHappen "good"

// define an event
let myEvent = Event<_>()

// define three observers
let observerA = fun i -> printfn "observer A noticed something, its value is %A" i
let observerB = fun i -> printfn "observer B noticed something, its value is %A" i
let observerC = fun i -> printfn "observer C noticed something, its value is %A" i

// publish the event and add observerA
myEvent.Publish   
|> Observable.add observerA

// publish the event and add observerA
myEvent.Publish 
|> Observable.add observerB

// publish the event and add observerA
myEvent.Publish 
|> Observable.add observerC

//fire event with value 1
myEvent.Trigger 1

