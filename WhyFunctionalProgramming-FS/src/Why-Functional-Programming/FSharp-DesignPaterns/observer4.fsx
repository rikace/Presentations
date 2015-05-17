// define an event
let myEvent = Event<_>()

// define three observers
let observerA = fun i -> printfn "observer A noticed something, its value is %A" i
let observerB = fun i -> printfn "observer B noticed something, its value is %A" i
let observerC = fun i -> printfn "observer C noticed something, its value is %A" i

// publish the event and add observerA
myEvent.Publish |> Observable.add observerA

// publish the event and add observerA
myEvent.Publish |> Observable.add observerB

// publish the event and add observerA
myEvent.Publish |> Observable.add observerC

//fire event wit h value 1
myEvent.Trigger 1