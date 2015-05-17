let add1 input = input + 1
let times2 input = input * 2

let concat2String input = (fst input) + (snd input) 

let genericLogger anyFunc input = 
   printfn "input is %A" input   //log the input
   let result = anyFunc input    //evaluate the function
   printfn "result is %A" result //log the result
   result                        //return the result

let add1WithLogging = genericLogger add1
let times2WithLogging = genericLogger times2
let concat2StringLogging = genericLogger concat2String

add1WithLogging 5

times2WithLogging 3

concat2StringLogging ("Functional Programming", " is cool!")

// define a adding function
let add x y = x + y

// normal use 
let z = add 1 2

let add42 = add 42


let genericLogger' before after anyFunc input = 
   before input               //callback for custom behavior
   let result = anyFunc input //evaluate the function
   after result               //callback for custom behavior
   result                     //return the result


let add1' input = input + 1

// reuse case 1
genericLogger' 
    (fun x -> printf "before=%i. " x) // function to call before 
    (fun x -> printfn " after=%i." x) // function to call after
    add1                              // main function
    2                                 // parameter 

// reuse case 2
genericLogger'
    (fun x -> printf "started with=%i " x) // different callback 
    (fun x -> printfn " ended with=%i" x) 
    add1                              // main function
    2                                 // parameter 

(*
This is a lot more flexible. 
I don't have to create a new function every 
time I want to change the behavior.
I can define the behavior on the fly.
*)

// define a reusable function with the "callback" functions fixed
let add1WithConsoleLogging = 
    genericLogger'
        (fun x -> printf "input=%i. " x) 
        (fun x -> printfn " result=%i" x)
        add1
        // last parameter NOT defined here yet! -> Partial Application

add1WithConsoleLogging 2
add1WithConsoleLogging 3
add1WithConsoleLogging 4