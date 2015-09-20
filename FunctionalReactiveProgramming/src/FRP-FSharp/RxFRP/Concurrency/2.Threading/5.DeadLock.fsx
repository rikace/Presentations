open System
open System.Threading

type BankAccount = { AccountID : int; OwnerName : string; mutable Balance : int }

let transferFunds amount fromAcct toAcct =
    printfn "Locking %s's account to deposit funds..." toAcct.OwnerName
    lock fromAcct
        (fun () ->
            printfn "Locking %s's account to withdraw funds..." fromAcct.OwnerName
            lock toAcct
                (fun () ->
                    fromAcct.Balance <- fromAcct.Balance - amount
                    toAcct.Balance   <- 
                    toAcct.Balance + amount
                )
        )

let john = { AccountID = 1; OwnerName = "John Smith"; Balance = 1000 }
let jane = { AccountID = 2; OwnerName = "Jane Doe";   Balance = 2000 }

ThreadPool.QueueUserWorkItem(fun _ -> transferFunds 100 john jane)
ThreadPool.QueueUserWorkItem(fun _ -> transferFunds 100 jane john)
