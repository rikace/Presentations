// State Pattern
// the interest rate is decided by the internal state: account balance

// define account state
type AccountState = 
    | Overdrawn
    | Silver
    | Gold

// define unit of measure as US dollar
[<Measure>] type USD

// define an account that takes the unit of measure
type Account<[<Measure>] 'u>() =
    // field to hold the account balance
    let mutable balance = 0.0<_>   

    // property for account state
    member this.State
        with get() = 
            match balance with
            | _ when balance <= 0.0<_> -> Overdrawn
            | _ when balance > 0.0<_> && balance < 10000.0<_> -> Silver
            | _ -> Gold

    // method to pay the interest
    member this.PayInterest() = 
        let interest = 
            match this.State with
                | Overdrawn -> 0.
                | Silver -> 0.01
                | Gold -> 0.02
        interest * balance

    // deposit into the account
    member this.Deposit x =  
        let a = x
        balance <- balance + a

    // withdraw from account
    member this.Withdraw x = 
        balance <- balance - x

// implement the state pattern
let state() = 
    let account = Account()

    // deposit 10000 USD
    account.Deposit 10000.<USD>

    // pay interest according to current balance
    printfn "account state = %A, interest = %A" account.State (account.PayInterest())

    // deposit another 2000 USD
    account.Withdraw 2000.<USD>

    // pay interest according to current balance
    printfn "account state = %A, interest = %A" account.State (account.PayInterest())

state()