open System
open System.IO

module ``Pipe Operaror`` =
    // Forward Pipe Operator
    // The Forward pipe operator is simply defined as:
    // let (|>) x f = f x
    //'a -> ('a -> 'b) -> 'b

    let multiBy2 x         = x * 2
    let toStr (x : int)  = x.ToString()
    let rev   (x : string) = new String(Array.rev (x.ToCharArray()))

    // 512 -> 1024 -> "1024" -> "4201"
    let result = rev (toStr (multiBy2 512))

    let result' = 512 |> multiBy2 |> toStr |> rev

module ``Pattern Matching`` =
    // Pattern matching is like a powerful switch statement, 
    // allowing you to branch control flow.
    // Pattern matching can also match against the structure of the data. 
    // So we can match against list elements joined together. 

    // A pattern match is a series of rules that will execute 
    // if the pattern matches the input. The pattern match expression 
    // then returns the result of the rule that was matched


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

module ``Discriminated Unions`` =
    // A discriminated union is a type that can 
    // only be one of a set of possible values. 
    
    // Each possible value of a discriminated union is 
    // referred to as a union case. 
    
    // With the invariant that discriminated unions can 
    // only be one of a set of values, 
    // the compiler can do additional checks to make 
    // sure your code is correct. 
     
    // The F# compiler ensures that pattern matches cover 
    // all discriminated union cases.

    // Discriminated union for a card's suit
    type Suit =
        | Heart
        | Diamond
        | Spade
        | Club

    let suits = [ Heart; Diamond; Spade; Club ]

    // Discriminated union for playing cards
    type PlayingCard =
        | Ace   of Suit
        | King  of Suit
        | Queen of Suit
        | Jack  of Suit
        | ValueCard of int * Suit

    // Use a list comprehension to generate a deck of cards.
    let deckOfCards =
        [
            for suit in [ Spade; Club; Heart; Diamond ] do
                yield Ace(suit)
                yield King(suit)
                yield Queen(suit)
                yield Jack(suit)
                for value in 2 .. 10 do
                    yield ValueCard(value, suit)
        ]

module ``Pattern Matching and Discriminated Unions`` = 
    open ``Discriminated Unions``

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

    // Pattern matching against types
    let whatIs (x : obj) =
        match x with
        | :? string    as s -> printfn "x is a string \"%s\"" s
        | :? int       as i -> printfn "x is an int %d" i
        | :? list<int> as l -> printfn "x is an int list '%A'" l
        | _ -> printfn "x is a '%s'" <| x.GetType().Name;

    whatIs "Hello Pattern Macthing"
    whatIs 7

module ``Active patterns`` =
    // are special functions that 
    // can be used inside of pattern-match rules. Using them 
    // eliminates the need for "when" guards as well as adding 
    // clarity to the pattern match. This has the extra benefit 
    // of making the code look as if it maps more closely to the 
    // problem you are solving.


    // Single-case active pattern

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


module UnitofMeasure = 
    // Units of measure allow you to pass along unit information
    [<Measure>]
    type fahrenheit

    let printTemperature (temp : float<fahrenheit>) =

        if   temp < 32.0<_>  then
            printfn "Below Freezing!"
        elif temp < 65.0<_>  then
            printfn "Cold"
        elif temp < 75.0<_>  then
            printfn "Just right!"
        elif temp < 100.0<_> then
            printfn "Hot!"
        else
            printfn "Scorching!"

    // Because the function only accepts fahrenheit values, 
    // it will fail to work with any floating-point values 
    // encoded with a different unit of measure

    let seattle = 59.0<fahrenheit>

    printTemperature seattle


    // ERROR: Different units
    [<Measure>]
    type celsius

    let nyc = 18.0<celsius>
    //printTemperature nyc


    // Define a measure for meters
    [<Measure>]
    type m

    // Multiplication, goes to meters squared
    // val it : float<m ^ 2> = 1.0
    1.0<m> * 1.0<m>

    // Division, drops unit entirely
    // val it : float = 1.0
    1.0<m> / 1.0<m>

    // Repeated division, results in 1 / meters
    1.0<m> / 1.0<m> / 1.0<m>

    [<Measure>]
    type mile = 
        static member asMeter = 1600.<m/mile>
    
    let d = 50.<mile> // Distance expressed using imperial units
    let d' = d * mile.asMeter // Same distance expressed using metric system
    
    printfn "%A = %A" d d'

// let error = d + d'       // Compile error: units of measure do not match


    
