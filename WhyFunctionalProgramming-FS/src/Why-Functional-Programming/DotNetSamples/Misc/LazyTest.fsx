
let getValueX() = 
        let x = 4
        printfn "Getting value x:%d" x 
        x

let getValueY() = 
        let y = 5
        printfn "Getting value y:%d" y 
        y

let addIftrue calculate x y =
    printfn "calculate is %b" calculate
    if calculate then 
        let result = x + y
        printfn "Add x:%d y:%d = %d" x y result
        result
    else 0


addIftrue true (getValueX()) (getValueY())
addIftrue false (getValueX()) (getValueY())

let addIftrueLazy calculate x y =
    printfn "calculate is %b" calculate
    if calculate then 
        let x' = x()
        let y' = y()
        let result = x' + y'
        printfn "Add x:%d y:%d = %d" x' y' result
        result
    else 0

addIftrueLazy true (getValueX) (getValueY)
addIftrueLazy false (getValueX) (getValueY)

let addIftrueVeryLazy calculate (x:Lazy<int>) (y:Lazy<int>) =
    printfn "calculate is %b" calculate
    if calculate then 
        let x' = x.Force()
        let y' = y.Force()
        let result = x' + y'
        printfn "Add x:%d y:%d = %d" x' y' result
        result
    else 0

addIftrueVeryLazy true (lazy getValueX()) (lazy getValueY())
addIftrueVeryLazy false  (lazy getValueX()) (lazy getValueY())

