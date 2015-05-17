module ChainOfResponsibility

// define a point record
type Point2D = {
    X : float
    Y : float
}

// create original point record
let originalPoint = { X = 0.0; Y = 0.0 }

// create (1,1) point record
let onePoint = { X = 1.0; Y = 1.0 }

// define a record to hold a person's age and weight
type Record = {
    Name : string;
    Age : int
    Weight: float
    Height: float
}

// Chain of responsibility pattern
let chainOfResponsibility() = 

    // function to check that the age is between 18 and 65
    let validAge record = 
        record.Age < 65 && record.Age > 18

    // function to check that the weight is less than 200
    let validWeight record = 
        record.Weight < 200.

    // function to check that the height is greater than 120
    let validHeight record = 
        record.Height > 120.

    // function to perform the check according to parameter f
    let check f (record, result) = 
        if not result then record, false
        else record, f(record)

    // create chain function
    let chainOfResponsibility = check validAge >> check validWeight >> check validHeight

    // define two patients' records
    let john = { Name = "John"; Age = 80; Weight = 180.; Height = 180. }
    let dan = { Name = "Dan"; Age = 20; Weight = 160.; Height = 190. }

    printfn "John's result = %b" (chainOfResponsibility (john, true) |> snd)
    printfn "Dan's result = %b" (chainOfResponsibility (dan, true) |> snd)

// chain template function
let chainTemplate processFunction canContinue s = 
    if canContinue s then 
        processFunction s
    else s

let canContinueF _ = true
let processF x = x + 1

//combine two functions to get a chainFunction
let chainFunction = chainTemplate processF canContinueF   

// use pipeline to form a chain
let s = 1 |> chainFunction |> chainFunction

printfn "%A" s

// define two units of measure: cm and kg
[<Measure>] type cm
[<Measure>] type kg

// define a person class with its height and weight set to 0cm and 0kg
type Person() = 
    member val Height = 0.<cm> with get, set
    member val Weight = 0.<kg> with get, set

// define a higher order function that takes a person record as a parameter
let makeCheck passingCriterion (person: #Person) = 
    if passingCriterion person then None  //if passing, say nothing, just let it pass
    else Some(person)   //if not passing, return Some(person) 

// define NotPassHeight when the height does not meet 170cm
let (| NotPassHeight | _ |) person = makeCheck (fun p -> p.Height > 170.<cm>) person

// define the NotPassWeight when weight does not fall into 100kg and 50kg range
let (| NotPassWeight | _ |) person = 
    makeCheck (fun p -> p.Weight < 100.<kg> && p.Weight > 50.<kg>) person

// check incoming variable x
let check x = 
    match x with
    | NotPassHeight x -> printfn "this person is not tall enough"
    | NotPassWeight x -> printfn "this person is out of weight range"
    | _ -> printfn "good, this person passes"

// create a person with 180cm and 75kg
let p = Person(Height = 180.<cm>, Weight = 75.<kg>)

// perform the chain check
check p

