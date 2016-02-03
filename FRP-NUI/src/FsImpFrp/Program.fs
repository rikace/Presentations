open FRPFSharp
open System
open EventModule

module FRP =
    type Time = float
    // Behaviors (signals) are flows of values, punctuated by event occurrences.
    type 'a Behavior = Beh of (Time -> 'a)
    type 'a Event = Evt of (Time * 'a) list  


type BankAccount() =

    let deposit = Event<int>.newDefault()

    let withdraw = Event<int>.newDefault()

    let accountEvents : Event<int> = deposit |> merge withdraw
    
    let bhAcc : Behavior<int> = accountEvents |> (accum 0 (+)) // Reevaluated for each update


    member x.Balance with get() = bhAcc.sample()
    member x.Deposit(amount) = deposit |> send(amount)
    member x.Withdorw(amount) = withdraw |> send(-amount) 

    // No Obsrevables - No Listeners / No Callbacks
    // No mutation of state 
    

[<EntryPoint>]
let main argv = 

    Console.ForegroundColor <- ConsoleColor.Green

    let bk = BankAccount()
    printfn "Initial Balance $%d\n" bk.Balance // ”Balance $0

    bk.Deposit(100)
    printfn "Balance $%d\n" bk.Balance // ”Balance $10    
    bk.Withdorw(25)    
    printfn "Balance $%d\n" bk.Balance // ”Balance $75”
    bk.Deposit(20)
    printfn "Balance $%d\n" bk.Balance // ”Balance $95”
    bk.Withdorw(70)
    printfn "Balance $%d\n" bk.Balance // ”Balance $95”

    Console.ReadLine() |> ignore
    printfn "%A" argv
    0 // return an integer exit code
