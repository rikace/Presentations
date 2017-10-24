open System

// --------------------------------------------------------
// Mutable and immutable values

let rnd = new System.Random()
let pair = ("Tomas", 42)
let inc x = x + 1

// Mutable values have to be marked explicitly
let mutable message = "Hello "
message <- message + "world!"
message <- message + "\n How are you?"
printfn "%s" message

// --------------------------------------------------------
// Introducing & decomposing F# tuples

let person = ("Tomas", 42)

let (name, age) = person
let name2 = fst person

let printPerson1 person =
    let (name,age) = person
    printfn "%s (%d)" name age

let printPerson2 (name,age) = 
    printfn "%s (%d)" name age

printPerson2 person


// age in range 0 to 150

let isValidAge (name, age) =
    if age >= 0 && age <= 150 then true
    else false

let printSafe (name, age) =
    if age < 18 then printfn "hidden"
    else printPerson1 (name, age)

let printSaf2 (name, age) =
    match age with
    | _ when age < 18 -> printfn "hidden"
    | _ -> printPerson1 (name, age)

let printSafe3 (name, age) =
    if age < 18 then printfn "hidden"
    else printPerson1 (name, age)

let isValidAge2 (name,age) =
    age >= 0 && age <= 150

let printSafe4 (name,age) =
    ignore((age > 18) && (printfn "%s" name; true))

// Takes a single parameter and decomposes
// it into name & age and adds 1 to the age
let addYear person = 
  let (name, age) = person
  (name, age + 1)
  
// Takes a single parameter, but the parameter
// is a tuple and is decomposed immediately!
let addYear2 (name, age) = (name, age + 1)

// --------------------------------------------------------
// Writing functions (in two ways)

let normalize (lo, hi) = 
  if (lo > hi) then (hi, lo)
  else (lo, hi)


let say message who = 
  printfn "%s %s!" message who

let sayHello = say "Hello"
sayHello "world"
sayHello "Tomas"

// --------------------------------------------------------
// Human development stages

let printHuman person = 
  match person with
  | (_, age) when age < 0 -> ()
  | (_, 0) -> ()
  | (_, age) when age < 18 -> ()
  | (_, 20) -> ()
  | 
  | _ -> printfn "Not implemented!"

// TODO: Implement 'printHuman' to do the following:
printHuman ("Alexander", 1)  // prints Infant
printHuman ("Joe", 15)       // prints Adolescent
printHuman ("Tomas", 42)     // prints Adult


// --------------------------------------------------------
// Implementing logic

let logicAnd (b1, b2) = 
  false // TODO: Implement logical and

for b1 in [ true; false ] do
  for b2 in [ true; false ] do
    if (logicAnd (b1, b2)) <> b1 && b2 then
      printfn "Error: %b and %b <> %b" b1 b2 (b1 && b2)


// TODO: Add test for 'or' and implement 'logicOr'

// --------------------------------------------------------
// Working with option types

let getPerson name =
  match name with
  | "Tomas" -> Some("Tomas", 42)
  | "Phil" -> Some("Phil", 18)
  | _ -> None


// TODO: Write a function that prints the result of 'getPerson' 
let printPerson info = 
   match (getPerson info) with
   | Some(n,a) -> printfn "%s %d" n a
   | None -> printfn "Not implemented!"
   // | None -> printfn "Not implemented!"

//printPerson (getPerson "Phil")
//printPerson (getPerson "Tomas")
//printPerson (getPerson "Alexander")
printPerson "Phil"
printPerson "Tomas"
printPerson "Alexander"






// Implementing three-value logic

let threeValueAnd b1 b2 = 
    match (b1, b2) with
    | Some(true), Some(true) -> Some(true)
    | Some(true), Some(false) | Some(false), _ | None, Some(false) -> Some(false)
    | None, _ | Some(true), None -> None
    //| _ -> Some(true)

let threeValueOr b1 b2 =
    match (b1, b2) with
    | Some(true), _ | Some(false), Some(true)  | None, Some(true) -> Some(true)
    | Some(false), Some(false) -> Some(false)
    | None, Some(false) | None, _ -> None
// TODO: Implement three value or
//    None

threeValueAnd (Some true) (Some false) = Some true
threeValueAnd (Some false) (Some false) = Some false
threeValueAnd (Some true) None = Some true
threeValueAnd (Some false) None = None

threeValueOr (Some true) (Some false) = Some true
threeValueOr (Some false) (Some false) = Some false
threeValueOr (Some true) None = Some true
threeValueOr (Some false) None = None

// --------------------------------------------------------
// Getting serious: Parsing formulas 

let tryParseFormula (s:string) =
  if s.StartsWith("=") then Some(s.Substring(1).Trim()) 
  else None

let tryParseNumber (s:string) = 
  match Int32.TryParse(s) with
  | true, num -> Some num
  | _ -> None

// TODO: Implement function that prints either
// "formula: %s" or "number: %d" depending on whether
// the input string is formula or number
let parser s = 
  printfn "nop"

parser "= 4 + 4"
parser "4"

// --------------------------------------------------------


//let(|ParseNumber|_|) text =
//    match Int32.TryParse(text) with
//    | true, num -> Some(sprintf "Number %d" num)
//    | _ -> None

let(|ParseNumber|_|) text =
    match Int32.TryParse(text) with
    | true, num -> Some(sprintf "Number %d" num)
    | _ -> None

let (|ParseExpression|_|) (text:string) =
    if text.StartsWith("=") then
        Some(sprintf "Formula %s" text)
    else None

let (|Lenght|) (text:string) =
    text.Length

let parseFormula text =
    match text with
    | Lenght 0 -> "Empty"
    | ParseNumber s -> s
    | ParseExpression s -> s
    | _ -> "Error"
