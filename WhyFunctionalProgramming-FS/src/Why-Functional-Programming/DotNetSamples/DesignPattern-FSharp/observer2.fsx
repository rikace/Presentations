
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

observer()



let task1 = async {
        let count = ref 0
        while true do
        do! Async.Sleep 1000
        printfn "task 1 .. %d" !count
        count := !count + 1
    }

let task2 = async {
    let count = ref 0
    while true do
    do! Async.Sleep 1000
    printfn "task 2 .. %d" !count
    count := !count + 1
}

let task3 = async {
        let count = ref 0
        while true do
        do! Async.Sleep 1000
        printfn "task 3 .. %d" !count
        count := !count + 1
    }

let cts = new System.Threading.CancellationTokenSource()

Async.Start(task1, cts.Token)
Async.Start(task2, cts.Token)
Async.Start(task3, cts.Token)

cts.Cancel()
