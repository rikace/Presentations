// The following sample wants to make sure the person’s age is between 18 and 65, 
// weight is no more than 200 and tall enough (>120).

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

chainOfResponsibility()
