open FRPFSharp
open System
open EventModule

module FRP =
    type Time = float
    // Behaviors (signals) are flows of values, punctuated by event occurrences.
    type 'a Behavior = Beh of (Time -> 'a)

    type 'a Event = Evt of (Time * 'a) list  


type BankAccount(initBalance) =

    let deposit = Event<int>.newDefault()

    let withdraw = Event<int>.newDefault()

    let accountEvents : Event<int> = deposit |> merge withdraw
    
    let bhAcc : Behavior<int> = accountEvents |> (accum initBalance (fun a b -> a + b)) // Reevaluated for each update

    member x.Balance with get() = bhAcc.sample()

    member x.Deposit(amount) = deposit |> send(amount)
  
    member x.Withdraw(amount) = withdraw |> send(-amount) 

    // No Obsrevables - No Listeners - No Callbacks
    // No mutation of state 
    // Events that create Behaviors
    // No need to register Events
    // No Need to un-register Events

[<EntryPoint>]
let main argv = 

    Console.ForegroundColor <- ConsoleColor.Green

    let bk = BankAccount(10)

    printfn "Initial Balance $%d\n" bk.Balance // ”Balance $10

    bk.Deposit(100)
    printfn "Balance $%d\n" bk.Balance // ”Balance $110

    bk.Withdraw(25)    
    printfn "Balance $%d\n" bk.Balance // ”Balance $85”

    bk.Deposit(20)
    printfn "Balance $%d\n" bk.Balance // ”Balance $105”

    bk.Withdraw(70)
    printfn "Balance $%d\n" bk.Balance // ”Balance $35”

    Console.ReadLine() |> ignore
    printfn "%A" argv
    0 // return an integer exit code
