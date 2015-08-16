//
//   ____   ____  _____  
//  / __ \ / __ \|  __ \ 
// | |  | | |  | | |__) |
// | |  | | |  | |  ___/ 
// | |__| | |__| | |     
//  \____/ \____/|_|     
//                       
//                       



module OOP 

    //*************************************************************************
    // OOP
    open System
    (*
        1. Define a class
        2. Members: Method, property
        3. Define an interface
        4. Abstract method
        5. Override a base method
        6. Upcasting with :>
        7. Downcast with :?>
        8. Type checking operation :?
    *)

    type IUser =
        abstract member GetUsername : unit -> string

    type BaseUser (username : string) =
        interface IUser with
            member this.GetUsername () =
                username


    let u = BaseUser("Riccardo") :> IUser

    let s = { new IUser with member this.GetUsername () = "Hello" }


    u.GetUsername()


    type Drawable =
        abstract member Draw: unit->unit

    type Shape(name) =
        let mutable _name = name

        interface Drawable with
            member this.Draw() =
                this.DoDraw()

        abstract member DoDraw:unit->unit
        default this.DoDraw()=
            printfn "drawing"

        member this.Describe =
            printfn "Sharp: %s" _name

        member this.Name 
            with get() =
                 _name
            and set value =
                _name<-value
    
    type Square(name, legs)=
        inherit Shape(name)
        let mutable _legs = legs

        member this.Legs
            with get() = _legs
            and set value = _legs<-value
        override this.DoDraw()=
            base.DoDraw()
            printfn "draw %s" name

            

    let aShape = new Shape("a")
    aShape.Describe
    printfn "%s" aShape.Name

    let square = new Square("square",4)
    printfn "legs: %d" square.Legs
    
    // Upcasting is checked at compile time, it will always successed at run time.
    // Upcasting happens automatically in methods.
    // The let-bound doesn't automatically upcasts.
    let drawable =square:>Drawable
    drawable.Draw()

    // Downcast is dynamic cast, it's checked at run time.
    // InvalidCastException will be thrown if failed.
    let squareBack = drawable:?>Square
    squareBack.Describe

    match drawable with
    | :? Square as square->
        square.Describe
    |_ -> ()

    // Delegate
    let countDigit (s:String) =
        [for x in s do if Char.IsDigit(x) then yield x].Length

    let count = countDigit "adfasdfas123456789"
    
    type countDigitDelegate = delegate of String -> int
    let dele = new countDigitDelegate(countDigit)
    let result = dele.Invoke("He110,w0r1d")


    

// Simple OO type (immutable):
type Person(forename : string, surname : string) =
   member __.Forename = forename
   member __.Surname = surname

// OO type with mutable members:
type MutablePerson(forename : string, surname : string) = 
   let mutable _forename = forename
   let mutable _surname = surname
   
   member this.Forename 
      with get () = _forename
      and set value = _forename <- value
   
   member this.Surname 
      with get () = _surname
      and set value = _surname <- value


// Mutable type using member val:
type MutablePerson2(forename : string, surname : string) = 
   member val Forename = forename with get, set
   member val Surname = surname with get, set

open System

// Mutable type with validation on creation (but not on mutation):
type SafePerson(forename : string, surname : string) = 
   
   let validateString str =
      if String.IsNullOrWhiteSpace str then
         raise (ArgumentException())
   do
      validateString forename
      validateString surname


   member val Forename = forename with get, set
   member val Surname = surname with get, set

// Mutable type with validation on creation and mutation:
type SafePerson2 (forename : string, surname : string) =
   let mutable _forename = forename
   let mutable _surname = surname

   let validateString str =
      if String.IsNullOrWhiteSpace str then
         raise (ArgumentException())
   do
      validateString forename
      validateString surname

   member this.Forename 
      with get () = _forename
      and set value = 
         validateString value
         _forename <- value
   member this.surname 
      with get () = _surname
      and set value = 
         validateString value
         _surname <- value


    // Interfaces:
    type IPerson =
       abstract member Forename : string
       abstract member Surname : string
       abstract member Fullname : string

    // Implementing an interface:
    type PersonFromInterface(forename : string, surname : string) =
       let validateString str =
          if String.IsNullOrWhiteSpace str then
             raise (ArgumentException())
       do
          validateString forename
          validateString surname

       interface IPerson with
          member __.Forename = forename
          member __.Surname = surname
          member __.Fullname = sprintf "%s %s" forename surname

    module InterfaceAccessDemo = 
       // Accessing interface members:
       let person = PersonFromInterface("Kit", "Eason")
       printfn "%s" (person :> IPerson).Fullname

    // For demonstrating access to Discriminated Union members from C# (See PersonTests.cs)
    type Contact =
       | PhoneNumber of AreaCode:string * Number:string
       | EmailAddress of string

    type ContactPerson(forename : string, surname : string, preferredContact : Contact) =
       let validateString str =
          if String.IsNullOrWhiteSpace str then
             raise (ArgumentException())
       do
          validateString forename
          validateString surname

       member __.PreferredContact = preferredContact

       interface IPerson with
          member __.Forename = forename
          member __.Surname = surname
          member __.Fullname = sprintf "%s %s" forename surname



    type IPoint = interface
        abstract X : int with get, set
        abstract Y : int with get, set
    end

    type Point2D() = class
        interface IPoint with
            member val X = 0 with get, set
            member val Y = 0 with get, set
    end

    type ``2DSurfaceWithOnePoint``(?upperBoundary:int, ?lowerBoundary:int) = class

        let upper = defaultArg upperBoundary 0
        let lower = defaultArg lowerBoundary 0

        let confine x = max (min x upper) lower

        /// lower boundary property
        member this.Upper with get() = upper

        /// upper boundary property
        member this.Lower with get() = lower

        /// Point property
        member val Point = Point2D() :> IPoint with get, set

        /// Move point'x value
        abstract member MoveX : int->unit
        default this.MoveX(delta:int) = 
            this.Point.X <- confine (this.Point.X + delta)

        /// Move point'y value
        abstract member MoveY : int->unit
        default this.MoveY(delta:int) = 
            this.Point.Y <- confine (this.Point.Y + delta)

        /// show the point's location
        member this.ShowLocation() = 
            printfn "current point location = (%d, %d)" this.Point.X this.Point.Y
    end



    type Person(first : string, last : string, age : int) =
        member p.FirstName = first
        member p.LastName = last
        member p.Age = age
        member p.SayHowdy() =
            System.Console.WriteLine("{0} says, 'Howdy, all'!", p.FirstName)
        override p.ToString() =
            System.String.Format("[Person: first={0}, last={1}, age={2}]",
                p.FirstName, p.LastName, age)
 
    let jess = new Person("Jessica", "Kerr", 33)
    jess.SayHowdy()
 



    type Student(first : string, last : string, age : int, subj : string) =
        inherit Person(first, last, age)
        let mutable subject = subj
        member s.Subject
            with get() = subject
            and set(value : string) = subject <- value
        override s.ToString() =
            System.String.Format("[Student: subject={0} {1}]", 
                s.Subject, base.ToString())
        member s.DrinkBeer(?amt : int, ?kind : string) =
            let beers = match amt with Some(beers) -> beers | None -> 12
            let brand = match kind with Some(name) -> name | None -> "Keystone"
            System.Console.WriteLine("{0} drank {1} {2}s!",
                s.FirstName, beers, brand)
 
    let brian = new Student("Brian", "Randell", 42, "Visual Basic")
    brian.DrinkBeer(6)
    brian.DrinkBeer(12, "Coors")
    brian.DrinkBeer()


