

// _______                    
// |__   __|                   
//    | |_   _ _ __   ___  ___ 
//    | | | | | '_ \ / _ \/ __|
//    | | |_| | |_) |  __/\__ \
//    |_|\__, | .__/ \___||___/
//        __/ | |              
//       |___/|_|              


open System
open System.IO

module Unit = 
    let sayHelloWorld () = printfn "Hello World"

    let add x y =
        let result = x + y
        () // inexplicably returning unit

    let add' x y =
        x + y |> ignore // inexplicably ignoring the result (ignore is a function that accepts T and returns unit)

    let ignore x = ()

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

module ``Record Types`` =

    // Records are a lightweight syntax for declaring a type with several 
    // public properties. 
    // One advantage of records is that by using the type 
    // inference system the compiler will figure out the type of the record 
    // by you simply setting its values. 
    
    // A record for a person's first and last name
    type Person =
        {FirstName:string; LastName:string}


        //override this.ToString() = sprintf "%s, %s" this.LastName this.FirstName


    let alice1 = {FirstName="Alice"; LastName="Adams"}
    let alice2 = {FirstName="Alice"; LastName="Adams"}
    let bob1 = {FirstName="Bob"; LastName="Bishop"}

    // structural equality
    let ``Is alice1 equals to alice2?`` =  alice1 = alice2


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

    type RecursiveEmployee = 
      | Worker of Person
      | Manager of RecursiveEmployee list
    
    
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
  //  sendEmail "bob@example.com"   //error


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
