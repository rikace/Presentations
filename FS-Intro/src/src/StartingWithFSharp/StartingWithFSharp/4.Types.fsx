

// _______                    
// |__   __|                   
//    | |_   _ _ __   ___  ___ 
//    | | | | | '_ \ / _ \/ __|
//    | | |_| | |_) |  __/\__ \
//    |_|\__, | .__/ \___||___/
//        __/ | |              
//       |___/|_|              


type User = { Username : string; Age : int }

let me = { Username = "bruinbrown93"; Age = 21 }

let GetAge user =
    user.Age



type PurchaseDeadline =
    | GivenDate of System.DateTime
    | EndOfDay


let purchase1 = EndOfDay
let purchase = GivenDate(System.DateTime.Now)


let tup = ("Test", 2)


let someNumber = Some 5
let noNumber = None


open System
open System.IO

// ============================
// Unit
// ============================
module Unit = 
    let sayHelloWorld () = printfn "Hello World"

    let add x y =
        let result = x + y
        () // inexplicably returning unit

    let add' x y =
        x + y |> ignore // inexplicably ignoring the result (ignore is a function that accepts T and returns unit)

    let ignore x = ()

    
// ============================
// Tuples
// ============================
module Tuples =

    // A tuple is an ordered collection of values treated like an atomic unit. 
    // A tuple allows you to keep things organized by grouping related values 
    // together without introducing a new type.
    let tuple = (1, false, "text")
    
    /// A simple tuple of integers
    let tuple1 = (1, 2, 3)

    /// A tuple consisting of an integer, a string, and a double-precision floating point number
    let tuple2 = (1, "fred", 3.1415)
        
    printfn "tuple1: %A    tuple2: %A" tuple1 tuple2

    /// A function that swaps the order of two values in a tuple.
    /// QuickInfo shows that the function is inferred to have a generic type.
    let swapElems (a, b) = (b, a)
    

    printfn "The result of swapping (1, 2) is %A" (swapElems (1, 2))



    
    let randomTuples = (1,true, "hello")

    let point = 1, 2 // int * int
    let x = fst point // get the first value
    let y = snd point // get the second value

    // fst and snd only work on "pairs" or tuples with only two vaues

    // A tuple pattern lets you assign 
    // explicit values from a tuple 
    let x', y' = point

    let anotherT = 1, "2", 3.0m // fst and snd won't work here.

    let _, _, z = anotherT
    
    let crazyTuple = (1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9)


// ============================
// Record Types
// ============================
module ``Record Types`` =

    // Records are a lightweight syntax for declaring a type with several 
    // public properties. 
    // One advantage of records is that by using the type 
    // inference system the compiler will figure out the type of the record 
    // by you simply setting its values. 
    
    // A record for a person's first and last name
    type Person =
        {FirstName:string; LastName:string}
        override this.ToString() = sprintf "%s, %s" this.LastName this.FirstName


    let alice1 = {FirstName="Alice"; LastName="Adams"}
    let alice2 = {FirstName="Alice"; LastName="Adams"}
    let bob1 = {FirstName="Bob"; LastName="Bishop"}

    //test
    printfn "alice1=alice2 is %A" (alice1=alice2)
    printfn "alice1=bob1 is %A" (alice1=bob1)


    let someone =
        {   FirstName = "Jeremy";
            LastName = "Abbott" }

    // copy everything and update first name
    let updateFirstName person firstName =
        { person with FirstName = firstName }

    let updatedPerson = updateFirstName someone "John"

    // Record type with a method on it
    type pointRecord = { XCoord : int; YCoord : int}
                        override x.ToString() =
                            sprintf "x coord: %i; y coord: %i" x.XCoord x.YCoord
                        member x.CalculateDistance(otherCoord : pointRecord) =
                            let xDiff = otherCoord.XCoord - x.XCoord
                            let yDiff = otherCoord.YCoord - x.YCoord
                            let result =  ((xDiff * xDiff) + (yDiff * yDiff))
                            int (sqrt (float result))

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


    //types can be combined recursively in complex ways
    type Person = {First:string; Last:string}
        
  //  type Person = Person of First:string * Last:string

    type Employee = 
      | Worker of Person
      | Manager of Employee list
    
    
    let jdoe = {First="John";Last="Doe"}
    let worker = Worker jdoe


        
    /// Define a discriminated union of 3 different kinds of employees
    type Employee = 
        /// Engineer is just herself
        | Engineer of Person
        /// Manager has list of reports
        | Manager of Person * list<Employee>
        /// Executive also has an assistant
        | Executive of Person * list<Employee> * Employee
    
    /// Find all managers/executives named "Dave" who do not have any reports
    let rec findDaveWithOpenPosition (emps : Employee list) = 
        emps |> List.filter (fun person -> // function 
                                match person with 
                                | Manager({ First = "Dave" }, []) -> true // [] matches the empty list
                                | Executive({ First = "Dave" }, [], _) -> true
                                | _ -> false) // '_' is a wildcard pattern that matches anything

 
    let engineer = Engineer({First="Riccardo"; Last="Terrell"} )
    let engineer' = Engineer({First="Mark"; Last="Quanq"} )
    let engineer'' = Engineer({First="Scott"; Last="Brown"} )
    let manager = Manager({First="Don"; Last="Syme"} , [engineer;engineer'])
    let manager' = Manager({First="Dave"; Last="Boo"} , [])
    let executive = Executive({First="Bugghina"; Last="Terrell"}, [engineer;engineer'],engineer'')

    findDaveWithOpenPosition [engineer; manager; executive]
    findDaveWithOpenPosition [engineer; manager; executive; manager']





    //define a "safe" email address type
    type EmailAddress = EmailAddress of string

    //define a function that uses it 
    let sendEmail (EmailAddress email) = 
       printfn "sent an email to %s" email

    //try to send one
    let aliceEmail = EmailAddress "alice@example.com"
    sendEmail aliceEmail

    //try to send a plain string
    sendEmail "bob@example.com"   //error





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


// This compiles down to an enumeration
type Color =
| Red = 0
| Blue = 1
| Green = 2

printfn "%i" (int Color.Red)

    // note that named union type fields is part of F# 3.1
    // prior to this named union types used tuple syntax

    // This compiles down to an abstract class with 
    // three inner classes inheriting from it
    type Shape =
    | Circle of Radius : float
    | Triangle of Base : float * Height : float
    | Rectangle of Length : float * Height : float
        member x.getArea () = 
            match x with // pattern matching
            | Circle (r) -> (r ** 2.0) * System.Math.PI 
            | Triangle (b, h) -> 0.5 * (b * h)
            | Rectangle (l, h) -> l * h


    let draw shape =    // define a function "draw" with a shape param
      match shape with
      | Circle radius -> 
          printfn "The circle has a radius of %f" radius
      | Rectangle (height,width) -> 
          printfn "The rectangle is %f high by %f wide" height width
      | _ -> printfn "I don't recognize this shape"


    let circle = Circle(10.)
    let rect = Rectangle(4.,5.)
    let triangle = Triangle(2.,3.)

    [circle; rect; triangle] |> List.iter draw


    let demoShape () =

        printfn "%s" "Discriminated Union (Shape) Info:"
        
        let circle = Shape.Circle(4.0)
        printfn "%A" (circle.getArea())

        let triangle = Shape.Triangle(3.0, 4.0)
        printfn "%A" (triangle.getArea())

        let rectangle = Shape.Rectangle(3.0, 4.0)
        printfn "%A" (rectangle.getArea())

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




    // types have built-in structural equality
    // types are automatically comparable
    type Suit = Club | Diamond | Spade | Heart
    type Rank = Two | Three | Four | Five | Six | Seven | Eight 
                | Nine | Ten | Jack | Queen | King | Ace
    
    let compareCard card1 card2 = 
        if card1 < card2 
        then printfn "%A is greater than %A" card2 card1 
        else printfn "%A is greater than %A" card1 card2 

    let aceHearts = Heart, Ace
    let twoHearts = Heart, Two
    let aceSpades = Spade, Ace

    compareCard aceHearts twoHearts 
    compareCard twoHearts aceSpades

    let hand = [ Club,Ace; Heart,Three; Heart,Ace; 
             Spade,Jack; Diamond,Two; Diamond,Ace ]

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

    // Exhaustive pattern matching

    
    type State = New | Draft | Published | Inactive | Discontinued

    let handleState state = 
       match state with
       | Inactive -> () // code for Inactive
       | Draft -> () // code for Draft
       | New -> () // code for New
       | Discontinued -> () // code for Discontinued





    // define a "union" of two different alternatives
    type Result<'a, 'b> = 
        | Success of 'a  // 'a means generic type. The actual type
                         // will be determined when it is used.
        | Failure of 'b  // generic failure type as well

    // define all possible errors
    type FileErrorReason = 
        | FileNotFound of string
        | UnauthorizedAccess of string * System.Exception

    // define a low level function in the bottom layer
    let performActionOnFile action filePath =
       try
          //open file, do the action and return the result
          use sr = new System.IO.StreamReader(filePath:string)
          let result = action sr  //do the action to the reader
          sr.Close()
          Success (result)        // return a Success
       with      // catch some exceptions and convert them to errors
          | :? System.IO.FileNotFoundException as ex 
              -> Failure (FileNotFound filePath)      
          | :? System.Security.SecurityException as ex 
              -> Failure (UnauthorizedAccess (filePath,ex))  
          // other exceptions are unhandled



    /// get the length of the text in the file
    let printLengthOfFile filePath = 
       let fileResult = 
         performActionOnFile (fun fs->fs.ReadToEnd().Length) filePath

       match fileResult with
       | Success result -> 
          // note type-safe int printing with %i
          printfn "length is: %i" result       
       | Failure _ -> 
          printfn "An error happened but I don't want to be specific"






    // -------- Object Expressions
    // Interfaces are useful, but sometimes you just want 
    // an implementation of an interface, without going
    // through the hassle of defining a custom type
    
    // Sorting a list using IComparer<'a>
    open System.Collections.Generic


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


