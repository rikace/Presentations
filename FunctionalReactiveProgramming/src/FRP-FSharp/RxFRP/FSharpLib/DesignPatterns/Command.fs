module Command

// define a command record
type Command = { Redo: unit->unit; Undo: unit->unit }

let commandPatternSample() = 

    // define a mutable storage
    let result = ref 7

    // define the add command
    let add n = { 
        Redo = (fun _ -> result := !result + n) 
        Undo = (fun _ -> result := !result - n) }

    // define the minus command
    let minus n = { 
        Redo = (fun _ -> result := !result - n) 
        Undo = (fun _ -> result := !result + n) }

    // define an add 3 command
    let cmd = add 3
    printfn "current state = %d" !result

    // perform add 3 redo operation
    cmd.Redo()
    printfn "after redo: %d" !result

    // perform an undo operation
    cmd.Undo()
    printfn "after undo: %d" !result

// define two command types
type CommandType = 
    | Deposit
    | Withdraw

// define the command format, which has a command type and an integer
type TCommand = 
    | Command of CommandType * int

// mutable variable result
let result = ref 7

// define a deposit function
let deposit x = result := !result + x

// define a withdraw function
let withdraw x = result := !result - x

// do function to perform a do action based on command type
let Do = fun cmd ->
    match cmd with
    | Command(CommandType.Deposit, n) -> deposit n
    | Command(CommandType.Withdraw,n) -> withdraw n

// undo function to perform an undo action based on command type
let Undo = fun cmd ->
    match cmd with
    | Command(CommandType.Deposit, n) -> withdraw n
    | Command(CommandType.Withdraw,n) -> deposit n

// print the current balance
printfn "current balance %d" !result

// deposit 3 into the account and print the balance
let depositCmd = Command(Deposit, 3)
Do depositCmd
printfn "after deposit: %d" !result

// undo the deposit command and print the balance
Undo depositCmd
printfn "after undo: %d" !result
