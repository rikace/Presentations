
    // “A type is a concept or abstraction, and is primarily about enforcing safety”
module ``Primitive Types`` = 
    //	Type inference   (Be careful not to confuse type inference with dynamic typing)
    
    //  The F# compiler doesn’t require you to explicitly state the types of all 
    //	the parameters to a function. The compiler infers their types based on usage.
    
    let sampleInteger = 176

    printfn "The value of sampleInteger is %d" sampleInteger // %s

    let answerToEverything = 42L
    
    let pi = 3.1415926M

    let bin = 0b00101010y

    let (b67:byte) = 67uy

    let bin' = "Polyglot DC UG"B
    
    /// Do some arithmetic starting with the first integer
    let sampleInteger2 = (sampleInteger / 4 + 5 - 7) * 4
    
    // strings
    let string1 = "Hello"
    let string2 = "world"
    let helloWorld = string1 + " " + string2 // concatenate the two strings with a space in between
    printfn "%s" helloWorld
    
    // Using a triple-quote string literal
    let string4 = """He said "hello world" 1261872 after you did"""
    
    /// A string formed by taking the first 7 characters of one of the result strings
    let substring = helloWorld.[0..6] //.Substring(0,6)
    printfn "%s" substring
    
    // immutablilty
    let x = 1
    x = x + 1 //?? does it make sense?


    let v = 2;  
    //v <- 3

    let mutable v2 = 2;
    v2 <- 3
    
    // Symbols
    let (!=)  x y = (x <> y)
    let (=/=) x y = (x != y)
 
module BasicFunctions = 
   
    // Use 'let' to define a function that accepts an integer argument and returns an integer.
    let func x = x * x + 3

    // Parenthesis are optional for function arguments
    let func' (x) = x * x + 3

    // int -> int 
    let square x = x * x
    
    // When can annotate the type of a parameter name using '(argument:type)'
    let someFunc (x : int) = 2 * x * x - x / 5 + 3
    let square' (x:int) : int = x * x
    
    // 	Because F# is statically typed, calling the add method 
    //	you just created with a floating-point value will result in a compiler error
    let add x y = x + y

    add 1 2
        
    //	But the + operator also works on floats too !?!?!?
    //  The reason is due to type inference. 
    //  Because the + operator works for many different types, such as byte, int, and decimal
    //	the compiler simply defaults to int if there is no additional information
    let inline add' x y = x + y  // The presence of inline affects type inference. 
                                 // This is because inline functions can have 
                                 // statically resolved type parameters

    let addOne = add 1 // partial appliaction
    let addTwo = add 2
    
    addOne 5

    
    /// Apply the function, naming the function return result using 'let'.
    /// The variable type is inferred from the function return type.
    let result = func' 4573
  
    printfn "The result of squaring the integer 4573 and adding 3 is %d" result 
        
    let printValue (value:int * int) (format:int * int-> unit) = format value

    printValue (2,2) (fun (x: int*int) -> printfn "Info: (%d, %d)" (fst x) (snd x))        
    printValue (3,3) (fun x -> printfn "Info: (%d, %d)" (fst x) (snd x))        
    printValue (4,4) (fun (x, y) -> printfn "Verbose: [%d------%d]" x y) // decompose the tuple
        
    // sample remove name and age and check signature
    let printName name age = printfn "My name is %s and I am %d" name age

    printName "Riccardo" 21


    let func3 x = 
        if x < 100.0 then 2.0 * x * x - x / 5.0 + 3.0
        else 2.0 * x * x + x / 5.0 - 37.0
    
    let result3 = func3 (6.5 + 4.5)
    
    printfn "The result of applying the 2nd sample function to (6.5 + 4.5) is %f" result3



     /// A list of all tuples containing all the numbers from 0 to 99 and their squares
    let sampleTableOfSquares = [ for i in 0..99 -> (i, i * i) ]
    
    // The next line prints a list that includes tuples, using %A for generic printing
    printfn "The table of squares from 0 to 99 is:\n%A" sampleTableOfSquares


    

   
// ----------------------------------------------------------------------------

    // Print the truth table for the given function
    let printTruthTable f =
        printfn "       |true   | false |"
        printfn "       +-------+-------+"
        printfn " true  | %5b | %5b |" (f true true)  (f true false)
        printfn " true  | %5b | %5b |" (f false true) (f false false)
        printfn "       +-------+-------+"
        printfn ""
        ()
    
    printTruthTable (&&)

    printTruthTable (||)

        // Compute the factorial of an integer. Use 'let rec' to define a recursive function    
    let rec factorial n =
        if n = 0 then 1
        else n * factorial (n - 1)

    printfn "%d" <| factorial 5
   

    // Define mutually recursive functions
    let rec isOdd  n = 
        if   n = 0 then false 
        elif n = 1 then true
        else isEven (n - 1)
    and isEven n = 
        if   n = 0 then true 
        elif n = 1 then false
        else isOdd (n - 1)

    let is5Odd = isOdd 5
    

    (*operator overload*)
    let rec (!) x =
        if x <= 1 then 1
        else x * !(x - 1)


    let (++) (x:int) (y:int) = x+2*y
    let b = 1 ++ 1


module Types = 

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


    let crazyTuple = (1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9)

// ----------------------------------------------------------------------------

    /// The class's constructor takes two arguments: dx and dy, both of type 'float'.
    type Vector2D(dx : float, dy : float) = 
        
        /// The length of the vector, computed when the object is constructed
        let length = sqrt (dx * dx + dy * dy)
        
        // 'this' specifies a name for the object's self identifier
        // In instance methods, it must appear before the member name.
        member this.DX = dx 
        member this.DY = dy
        member this.Length = length         
        member this.Scale(k) = Vector2D(k * this.DX, k * this.DY)
    
    /// An instance of the Vector2D class
    let vector1 = Vector2D(3.0, 4.0)
    
    /// Get a new scaled vector object, without modifying the original object
    let vector2 = vector1.Scale(10.0)
    
    printfn "Length of vector1: %f      Length of vector2: %f" vector1.Length vector2.Length


    // Generic classes
    // Define a generic class
    type Arrayify<'a>(x : 'a) =
        member this.EmptyArray : 'a[] = [| |]
        member this.ArraySize1 : 'a[] = [| x |]
        member this.ArraySize2 : 'a[] = [| x; x |]
        member this.ArraySize3 : 'a[] = [| x; x; x |]
    
    let arrayifyTuple = new Arrayify<int * int>( (10, 27) )
    
    arrayifyTuple.ArraySize3
 
    //Generated Equality  
    type ClassType(x : int) =
        member this.Value = x

    let x = new ClassType(31)
    let y = new ClassType(31)

    x = y


    // Tuples, discriminated unions, and records behave exactly like you would expect
    // two instances with the same set of values are considered equal, 
    // just like value types.

    let tupleX = ('a', 2)
    let tupleY = ('a', 2)

    tupleX = tupleY


    // Records are a lightweight syntax for declaring a type with several 
    // public properties. 
    // One advantage of records is that by using the type 
    // inference system the compiler will figure out the type of the record 
    // by you simply setting its values. 
    
    // A record for a person's first and last name
    type Person =
        { First : string
          Last : string }
        override this.ToString() = sprintf "%s, %s" this.Last this.First
    
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


    //	Record Type 
    type PersonalInfo = { Name : string; Id : int }

    let spi1 = {Name="Scott"; Id=0}
    let spi2 = {Name="Scott"; Id=0}

    spi1 = spi2


    // Records are immutable, they can easily be cloned using the with keyword:
    type Car =
        {
            Make  : string
            Model : string
            Year  : int
        }
    let thisYear = { Make = "FSharp"; Model = "Luxury Sedan"; Year = 2012 }
    let nextYear = { thisYear with Year = 2013 }


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


    // -------- Extension methods 
    // provide a simple extension mechanism without
    // needing to modify a type hierarchy or modify existing types.

    // Extend the System.Int32 AKA int type
    type System.Int32 with
        member this.ToHexString() = sprintf "0x%x" this

    (1094).ToHexString();


    [<Struct>]
    type StructPoint(x : int, y : int) =
        member this.X = x
        member this.Y = y
    let dinner = ("green egg", "ham")



