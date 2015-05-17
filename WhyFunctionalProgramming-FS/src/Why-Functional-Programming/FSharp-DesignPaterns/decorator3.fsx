type Divide() = 
    // define basic divide function
    let mutable divide = fun (a,b) -> a / b

    // define a property to expose the function
    member this.Function
        with get() = divide
        and set(v) = divide <- v

    // method to invoke the function
    member this.Invoke(a,b) = divide (a,b)

// decorator pattern
let decorate() = 

    // create a divide instance
    let d = Divide()

    // set the check zero function
    let checkZero (a,b) = if b = 0 then failwith "a/b and b is 0" else (a,b)
    

    // invoke the function without check zero
    try 
        d.Invoke(1, 0) |> ignore
    with e -> printfn "without check, the error is = %s" e.Message

    // add the check zero function and then invoke the divide instance
    d.Function <- checkZero >> d.Function 
    try
        d.Invoke(1, 0) |> ignore
    with e -> printfn "after add check, error is = %s" e.Message

decorate()