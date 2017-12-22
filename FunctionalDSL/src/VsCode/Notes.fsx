open System
open System.Net
open System.Windows
open System.Threading



let a&b = 3 // create 2 variable a & b with both value 3

[<LiteralAttribute>]
let S = "Riccardo"

let f error =
    try
        failwith error
    with
    | :? ArgumentException as a -> ()
    | :? Exception as e -> ()


let arr = [| Some(1); None; Some(2); Some(3); None |]
arr |> Array.choose(fun x -> x) // return the Value of option Some

let m = Map.empty
let ms = Map.ofList[ 1, "Riccardo"; 2, "Bugghina" ]

////////////////////////////////////////////////////////////////////////
let (|Contains|) (search:string, value:string) =
    search.Contains value

let test value =
    match value with
    | Contains true -> sprintf "yes"
    | Contains false -> sprintf "no"

test ("riccardo", "cc")
////////////////////////////////////////////////////////////////////////

let (|Match|_|) pat value =
    let results = (System.Text.RegularExpressions.Regex.Matches(value, pat))
    if results.Count > 0 then Some [for r in results -> r.Value]
    else None

let matchValue value = 
    match value with
    |Match "test" results -> sprintf "yes"
    |_ -> sprintf "no"
////////////////////////////////////////////////////////////////////////


let (|FTP|HTTP|HTTPS|UNKOWON|) (value:string) =
    match value.ToLower() with
    | "ftp" -> FTP value
    | "http" -> HTTP
    | "https" -> HTTPS
    | _ -> UNKOWON

let testUri uri =
    match uri with
    | FTP v -> sprintf "FTP %s" v
    | HTTP -> sprintf "HTTP"
    | _ -> "ops"
////////////////////////////////////////////////////////////////////////

enum<System.DayOfWeek>(3) // Wednesday

////////////////////////////////////////////////////////////////////////

[<System.FlagsAttribute>]
type Soda =
    | Coke = 0x0001
    | Fanta = 0x0002
    | Ginger = 0x0004

let cokeAndFanta = Soda.Coke ||| Soda.Fanta

let rec getDrink soda =
    match soda with
    | Soda.Coke -> sprintf "Only Coke"
    | _ when (soda &&& Soda.Coke = Soda.Coke) -> printfn "Coke"
                                                 getDrink (soda ^^^ Soda.Coke)
    | _ when (soda &&& Soda.Fanta = Soda.Fanta) -> sprintf "Fanta"
    | _ when (soda &&& Soda.Ginger = Soda.Ginger) -> sprintf "Ginger"

getDrink cokeAndFanta
////////////////////////////////////////////////////////////////////////

