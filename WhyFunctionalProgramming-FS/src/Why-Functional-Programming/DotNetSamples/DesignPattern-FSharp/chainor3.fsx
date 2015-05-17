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

