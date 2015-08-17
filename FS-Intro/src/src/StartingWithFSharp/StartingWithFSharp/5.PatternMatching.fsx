


//  _____      _   _                                    _       _     
// |  __ \    | | | |                                  | |     | |    
// | |__) |_ _| |_| |_ ___ _ __ _ __    _ __ ___   __ _| |_ ___| |__  
// |  ___/ _` | __| __/ _ \ '__| '_ \  | '_ ` _ \ / _` | __/ __| '_ \ 
// | |  | (_| | |_| ||  __/ |  | | | | | | | | | | (_| | || (__| | | |
// |_|   \__,_|\__|\__\___|_|  |_| |_| |_| |_|_|_|\__,_|\__\___|_| |_|


//           _ _   _   _            _   _     _
//     /\   | | | | | | |          | | | |   (_)                      
//    /  \  | | | | |_| |__   ___  | |_| |__  _ _ __   __ _ ___       
//   / /\ \ | | | | __| '_ \ / _ \ | __| '_ \| | '_ \ / _` / __|      
//  / ____ \| | | | |_| | | |  __/ | |_| | | | | | | | (_| \__ \      
// /_/    \_\_|_|  \__|_| |_|\___|  \__|_| |_|_|_| |_|\__, |___/      
//                                                     __/ |          
//                                                    |___/           


REGEX

match tup with
| ("Test", _) -> printfn "Success"
| (_, 1)
| (_, 2) -> printfn "Small number"
| _ -> printfn "Bad idea"

let parse = false

match tup, parse with
| _, false -> printfn "Not parsing"
| _, true -> printfn "Parsing"



match purchase with
| EndOfDay -> printfn "End of day"
| GivenDate date -> printfn "%A" date



match me with
| { Username = "bruinbrown93"; Age = _ } -> printfn "Cool guy"
| _ -> printfn "Have you met bruinbrown93?"



let numbers = [ 10; 20; 30; 40 ]

match numbers with
| [ w; x; y; z] -> printfn "numbers had 4 elements which sum to %i" (w + x + y + z)
| _ -> printfn "Not 4 numbers"


match numbers with
| x :: xs -> printfn "Head was %i" x
| _ -> printfn "Empty list"



let (|Even|Odd|) a =
    if (a % 2) = 0 then Even else Odd


match 5 with
| Even -> printfn "Even"
//| Odd -> printfn "Odd"


module PatternMatching


// *********************************
// Basic expression pattern matching
// *********************************

//matching tuples directly
let first, second, _ =  (1,2,3)  // underscore means ignore

//matching lists directly
let e1::e2::rest = [1..10]       // ignore the warning for now

//matching lists inside a match..with
let listMatcher aList = 
    match aList with
    | [] -> printfn "the list is empty" 
    | [first] -> printfn "the list has one element %A " first 
    | [first; second] -> printfn "list is %A and %A" first second 
    | _ -> printfn "the list has more than two elements"

listMatcher [1;2;3;4]
listMatcher [1;2]
listMatcher [1]
listMatcher []




let printMatchedExpression result =
    printfn "%s" result

// Default pattern matching syntax
let getCodeValue code =
    match code with
        | "A" -> "Awesome"
        | "B" -> "Best"
        | "C" -> "Common"
        | _ -> "Unknown Code Type" // Wild card pattern

    
// Short hand pattern matching syntax
let getCodeValue' =
    function
        | "A" -> "Awesome"
        | "B" -> "Best"
        | "C" -> "Common"
        | _ -> "Unknown Code Type" // Wild card pattern

// Short hand pattern matching syntax
let getCodeValue'' =
    function
        | "A" -> "Awesome"
        | "B" -> "Best"
        | "C" -> "Common"
        | c -> sprintf "Code: %s was input, but does not have a known value" c // Wild card pattern

let printBasicPatternMatchingExample () = 
    printMatchedExpression (getCodeValue "A")
    printMatchedExpression (getCodeValue "Z")

    printMatchedExpression (getCodeValue' "A")
    printMatchedExpression (getCodeValue' "Z")

    printMatchedExpression (getCodeValue'' "A")
    printMatchedExpression (getCodeValue'' "Z")




// Some(..) and None are roughly analogous to Nullable wrappers
let validValue = Some(99)
let invalidValue = None

// In this example, match..with matches the "Some" and the "None",
// and also unpacks the value in the "Some" at the same time.
let optionPatternMatch input =
   match input with
    | Some i -> printfn "input is an int=%d" i
    | None -> printfn "input is missing"

optionPatternMatch validValue
optionPatternMatch invalidValue



// *********************************
// Pattern matching against a tuple
// *********************************
let points = [0, 0; 1, 0; 0, 1; -2, 3] 
let locatePoint p =
    match p with
    | (0, 0) -> sprintf "%A is at the origin" p
    | (x, 0) when x >= 1  -> sprintf "%A is on the x-axis" p
    | (0, _) -> sprintf "%A is on the y-axis" p
    | (x, y) -> sprintf "%A is at x: %i, y: %i" p x y
    
let printTuplePatternMatchingExample () =
    points |> List.map locatePoint |> List.iter (fun s -> printfn "%s" s)

// *********************************
// Pattern matching against Record Types
// *********************************
type Model =
    | Six
    | SixPlus
    | Five
    | FiveS
type Phone = { Manufacturer : string; Model : Model; OperatingSystem : string; Storage : int }
let phones = [{ Manufacturer = "Apple"; Model = Model.Six; OperatingSystem = "iOS"; Storage = 64 };
                { Manufacturer = "Apple"; Model = Model.Six; OperatingSystem = "iOS"; Storage = 128 };
                { Manufacturer = "Apple"; Model = Model.SixPlus; OperatingSystem = "iOS"; Storage = 64 };
                { Manufacturer = "Apple"; Model = Model.Five; OperatingSystem = "iOS"; Storage = 64 }]
let isNewEnough =
    function
    | { Model = Model.SixPlus } -> true
    | { Model = Model.Six } -> true
    | _ -> false
let printRecordTypePatternMatching () =
    phones
    |> List.filter isNewEnough
    |> List.iter
        (fun p -> printfn "This phone is new enough - Manufacturer: %s, Storage: %i" p.Manufacturer p.Storage)    