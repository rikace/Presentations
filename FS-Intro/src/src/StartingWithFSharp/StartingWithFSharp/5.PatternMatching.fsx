


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


open System
open System.IO                 
                         
module ``Pattern Matching`` =
    // Pattern matching is like a powerful switch statement, 
    // allowing you to branch control flow.
    // Pattern matching can also match against the structure of the data. 
    // So we can match against list elements joined together. 

    // A pattern match is a series of rules that will execute 
    // if the pattern matches the input. The pattern match expression 
    // then returns the result of the rule that was matched

    let function1 x =
        match x with
        | (var1, var2) when var1 > var2 -> printfn "%d is greater than %d" var1 var2 
        | (var1, var2) when var1 < var2 -> printfn "%d is less than %d" var1 var2
        | (var1, var2) -> printfn "%d equals %d" var1 var2

    function1 (1,2)
    function1 (2, 1)
    function1 (0, 0)


    let (var1, var2) as tuple1 = (1, 2)
    printfn "%d %d %A" var1 var2 tuple1
    

    type PersonName =
        | FirstOnly of string
        | LastOnly of string
        | FirstLast of string * string

    let constructQuery personName = 
        match personName with
                | FirstOnly(firstName) -> printfn "May I call you %s?" firstName
        | LastOnly(lastName) -> printfn "Are you Mr. or Ms. %s?" lastName              
        | FirstLast(firstName, lastName) -> printf "Are you %s %s?" firstName lastName


    constructQuery (FirstOnly "Riccardo")



    let isOdd x = (x % 2 = 1)   

    let describeNumber x =
        match isOdd x with
        | true  -> printfn "x is odd"
        | false -> printfn "x is even"

    describeNumber 7


    


    let highLowGame () =
        let rng = new Random()
        let secretNumber = rng.Next() % 100

        let rec highLowGameStep () =

            printfn "Guess the secret number:"
            let guessStr = Console.ReadLine()
            let guess = Int32.Parse(guessStr)

            match guess with
            | _ when guess > secretNumber
                -> printfn "The secret number is lower."
                   highLowGameStep()

            | _ when guess = secretNumber
                -> printfn "You've guessed correctly!"
                   ()

            | _ when guess < secretNumber
                -> printfn "The secret number is higher."
                   highLowGameStep()
        highLowGameStep()

    // Begin the game
    highLowGame()


module ``Pattern Matching and Discriminated Unions`` = 


// You can pattern match against discriminated unions by using
// just the case labels as patterns. 

// If the union label has data associated with it, 
// then you can match its value against 
// a constant, a wildcard, or capture the value just 
// like a normal pattern match.

// Pattern matching adds power to your programming by giving 
// you a way to be much more expressive in code branching than 
// using if expressions alone. It allows you to match against constants, 
// capture values, and match against the structure of data.


    // types have built-in structural equality
    // Discriminated union for a card's suit
    type Suit =
        | Heart
        | Diamond
        | Spade
        | Club

    let suits = [ Heart; Diamond; Spade; Club ]

    // Discriminated union for playing cards
    type PlayingCard =
        | ValueCard of int * Suit
        | King  of Suit
        | Queen of Suit
        | Jack  of Suit
        | Ace   of Suit

 
    /// Converts a 'Card' object to a string
    let suitString (s:Suit) = 
        match s with
        | Club -> "Clubs"
        | Diamond -> "Diamonds"
        | Spade -> "Spades"
        | Heart -> "Hearts"

    // string -> Suitf
    let stringToSuit s = 
        match s with
        | "Clubs" -> Club
        | "Diamonds" -> Diamond
        | "Spades" -> Spade
        | "Heart" -> Heart
         
    let rankString c = 
        match c with
        | Ace(s) -> printfn "Ace of %s" (suitString s)
        | King(s) -> printfn "King of %s" (suitString s)
        | Queen(s) -> printfn "Queen of %s" (suitString s)
        | Jack(s) ->  printfn "Jack of %s" (suitString s)
        | ValueCard(n, s) ->  printfn "Card %d of %s" n (suitString s)
    
    rankString (Ace(Spade))
    rankString (King(Club))
    
    // Describe a pair of cards in a game of poker
    let describeHoleCards cards =
        match cards with
        | []
        | [_] -> failwith "Too few cards."
        | cards when List.length cards > 2
            -> failwith "Too many cards."

        | [ Ace(_);  Ace(_)  ] -> "Pocket Rockets"
        | [ King(_); King(_) ] -> "Cowboys"

        | [ ValueCard(2, _); ValueCard(2, _)]
            -> "Ducks"

        | [ Queen(_); Queen(_) ]
        | [ Jack(_);  Jack(_)  ] -> "Pair of face cards"

        | [ ValueCard(x, _); ValueCard(y, _) ] when x = y
            -> "A Pair"

        | [ first; second ]
            -> sprintf "Two cards: %A and %A" first second

    describeHoleCards [Ace(Diamond); King(Spade)]
    describeHoleCards [Queen(Heart);ValueCard(7, Club)]

    



     
    let compareCard card1 card2 = 
        if card1 < card2 
        then printfn "%A is greater than %A" card2 card1 
        else printfn "%A is greater than %A" card1 card2 

    let aceHearts = Ace(Heart)
    let twoHearts = ValueCard(2, Heart)
    let aceSpades = Ace(Spade)

    compareCard aceHearts twoHearts 
    compareCard twoHearts aceSpades

    let hand = [ ValueCard(3, Spade); Ace(Heart);
                 Jack(Spade); ValueCard(2, Diamond); Ace(Diamond) ]

    //instant sorting!
    List.sort hand |> printfn "sorted hand is (low to high) %A"
    
    List.max hand |> printfn "high card is %A"
    List.min hand |> printfn "low card is %A"




    // Pattern matching against types
    let whatIs (x : obj) =
        match x with
        | :? string    as s -> printfn "x is a string \"%s\"" s
        | :? int       as i -> printfn "x is an int %d" i
        | :? list<int> as l -> printfn "x is an int list '%A'" l
        | _ -> printfn "x is a '%s'" <| x.GetType().Name;

    whatIs "Hello Pattern Macthing"
    whatIs 7




    // -------- Object Expressions
    // Interfaces are useful, but sometimes you just want 
    // an implementation of an interface, without going
    // through the hassle of defining a custom type
    
    // Sorting a list using IComparer<'a>
    open System.Collections.Generic

    type Person = {First:string; Last:string}
        with
            override x.ToString() = sprintf "%s %s" x.First x.Last 

    let people =
        new List<_>(
            [|
                { First = "Jomo";  Last = "Fisher" }
                { First = "Brian"; Last = "McNamara" }
                { First = "Joe";   Last = "Pamer" }
            |] )

    let printPeople ()  =
        Seq.iter (fun person -> printfn "\t %s" (person.ToString())) people

    printPeople()
    // Sort people by first name
    people.Sort({   new IComparer<Person> with
                        member this.Compare(l, r) =
                            if   l.First > r.First then  1
                            elif l.First = r.First then  0
                            else                        -1     })

    printPeople()

module ``Active patterns`` =
    // are special functions that 
    // can be used inside of pattern-match rules. Using them 
    // eliminates the need for "when" guards as well as adding 
    // clarity to the pattern match. This has the extra benefit 
    // of making the code look as if it maps more closely to the 
    // problem you are solving.



    let (|Even|Odd|) a =
        if (a % 2) = 0 then Even else Odd


    match 5 with
    | Even -> printfn "Even"
    //| Odd -> printfn "Odd"

    // Convert a file path into its extension
    let (|FileExtension|) filePath = System.IO.Path.GetExtension(filePath)

    let determineFileType (filePath : string) =
        match filePath with

        // Without active patterns
        | filePath when Path.GetExtension(filePath) = ".txt"
            -> printfn "It is a text file."

        // Converting the data using an active pattern
        | FileExtension ".jpg"
        | FileExtension ".png"
        | FileExtension ".gif"
            -> printfn "It is an image file."

        // Binding a new value
        | FileExtension ext
            -> printfn "Unknown file extension [%s]" ext


    // Partial Active Pattern
    // Active pattern for converting strings to ints
    let (|ToInt|) x = System.Int32.Parse(x)

    let (|KB|MB|GB|) filePath =
        let file = System.IO.File.Open(filePath, System.IO.FileMode.Open)
        if file.Length < 1024L * 1024L then
            KB
        elif file.Length < 1024L * 1024L * 1024L then
            MB
        else GB

    let (|IsImage|_|) filePath =
        let ext = System.IO.Path.GetExtension(filePath)
        match ext with
        | ".jpg" 
        | ".bmp"
        | ".gif" -> Some()
        | _ -> None

    let BigImage filePath =
        match filePath with
        | IsImage & (MB | GB) -> true
        | _ -> false

    BigImage @""

    // Check if the input string parses as the number 4
    let isFour str =
        match str with
        | ToInt 4 -> true
        | _ -> false

    isFour " 4 "



    let (|ToBool|_|) x =
        let success, result = Boolean.TryParse(x)
        if success then Some(result)
        else            None

    let (|ToInt|_|) x =
        let success, result = Int32.TryParse(x)
        if success then Some(result)
        else            None

    let (|ToFloat|_|) x =
        let success, result = Double.TryParse(x)
        if success then Some(result)
        else            None

    let describeString str =
        match str with
        | ToBool  b -> printfn "%s is a bool with value %b" str b
        | ToInt   i -> printfn "%s is an integer with value %d" str i
        | ToFloat f -> printfn "%s is a float with value %f" str f
        | _         -> printfn "%s is not a bool, int, or float" str

    describeString " 3.141 "
    describeString "Not a valid integer"


    // GetFolderSize.fsx sample



