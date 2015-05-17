// implement the decorate pattern in F#. The decorate pattern is to add new featuures to an object at runtime.

type Divide() = 
    let mutable divide = fun (a,b) -> a / b
    member this.Function
        with get() = divide
        and set(v) = divide <- v
    member this.Invoke(a,b) = divide (a,b)

let decorate() = 
    let d = Divide()
    let checkZero (a,b) = if b = 0 then failwith "a/b and b is 0" else (a,b)
    
    try 
        d.Invoke(1, 0) |> ignore
    with e -> printfn "without check, the error is = %s" e.Message

    d.Function <- checkZero >> d.Function 
    try
        d.Invoke(1,0) |> ignore
    with e -> printfn "after add check, error is = %s" e.Message

decorate()