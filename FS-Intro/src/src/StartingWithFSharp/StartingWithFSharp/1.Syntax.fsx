open System
open System.IO

// single line comments use a double slash
(* multi line comments use (* . . . *) pair

-end of multi line comment- *)

// =====================================
// ========     Basic Syntax    ========
// =====================================
module ``Basic Syntax`` = 
    // The "let" keyword defines an (immutable) value
    let myInt = 5
    let myFloat = 3.14
    let myString = "hello" //note that no types needed    
    let tuple = myInt, myString // pack into tuple   
    let s2, i2 = tuple // unpack
    let list = [ s2 ] // type is string list
    
    // The printf/printfn functions are similar to the
    // Console.Write/WriteLine functions in C#.
    printfn "Printing an int %i, a float %f, a bool %b" 1 2.0 true
    printfn "A string %s, and something generic %A" "hello" [ 1; 2; 3; 4 ]

// =====================================
// ========     Immutability    ========
// =====================================
module Immutability = 
    let x = 1
    
    x = x + 1 // ?? does it make sense?
    
    // What's the value of x ??
    // Bind the value 2 to the name "y"
    let y = 2 // value x : int = 2
    
    y = 3 // val it : bool = false
    y = 2 // val it : bool = true
    
    let v = 2
    //v <- 3
    let mutable v2 = 2
    
    v2 <- 3
    
    // immutable list
    let immutableList = [ 1; 2; 3; 4 ]
    
    // The "equal" sign is used for bindings and for comparison
    // Make something mutable
    let mutable z = 2
    
    z = 2 // val it : bool = true
    z = 3 // val it : bool = false
    z <- 3 // This is how you assign a value to a mutable binding
    z = 3 // val it : bool = true

// =====================================
// ==========      Alias     ===========
// =====================================
module Alias = 
    // I find it useful to describe the Domain Model (DDD)
    type ProductCode = string
    
    type transform<'a> = 'a -> 'a
    
    type RealNumber = float
    
    type ComplexNumber = float * float
    
    type CustomerId = int
    
    type AdditionFunction = int -> int -> int
    
    type ComplexAdditionFunction = ComplexNumber -> ComplexNumber -> ComplexNumber
    
    // Great for DSL
    type FirstName = string
    
    type LastName = string
    
    let createPeraon (firstName : FirstName, lastName : LastName) = 
        ()


    // create person
    let (!=) // Symbols
             x y = (x <> y)

    let (=/=) x y = (x != y)

    let x = 5
    let y = 4
    
    let ``are x and y equal`` = 
        if x <> y then false
        elif x != y then false
        elif x =/= y then false
        else true
