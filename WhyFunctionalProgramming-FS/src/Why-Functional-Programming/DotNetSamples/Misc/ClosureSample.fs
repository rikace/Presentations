namespace Misc

(*
All functional languages include closures. 
A closure is a function that carries an implicit binding to all the variables referenced within it. 
In other words, the function encloses a context around the things it references.
*)

module closure =
// clousre sample in F#
// The makeCounter() function first defines a local variable with an appropriate name, 
// then returns a code block that uses that variable.
    
    let makeCounter() =
        let localVal = ref 0

        let makeCounter() =
            localVal := !localVal + 1
            !localVal 
        makeCounter


    // c1 now points to an instance of the code block
    // calling c1 increments the internal variable
    let c1 = makeCounter()
    c1()
    c1()
    c1()

    // c2 now points to a new, unique instance of makeCounter()
    let c2 = makeCounter()

    printfn "C1 = %d, C2 = %d" (c1()) (c2()) // output: C1 = 4, C2 = 1  

// each of the code blocks has kept track of a separate instance of localVal. 
// closures operate by enclosing context