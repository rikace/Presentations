// Learn more about F# at http://fsharp.net

module Composition =
    let add x y = x + y
    let mult x y = x * y
    let square x = x * x

    let add' = fun x y -> x + y
    // (int -> int -> int)
    let add'' x = fun y -> x + y
    // int -> (int -> int)

    let add10'' = add'' 10
    printfn "%d" (add10'' 32)

    let add10 = add 10
    printfn "%d" (add10 32)

    let addmultipleValues a b c d = a + b + c + d

    let result = addmultipleValues 10

    let result2 = result 15

    let mult5 = mult 5

    let calcResult = mult5 (add10 17)
    let calcResult' = 17 |> add10 |> mult5

    let add10mult5 = add10 >> mult5

    let calcResult'' = add10mult5 17


module DataManipulationAndAnalysis =

    type Product = { Name: string; Price: decimal }
    type OrderLine = { Product: Product; Count: int }
    type Order = { OrderId: string; Lines: OrderLine list }

    let rubberChicken = { Name = "Rubber chicken"; Price = 8.99m }
    let pulley = { Name = "Pulley"; Price = 1.95m }
    let fairyDust = { Name = "Fairy Dust"; Price = 3.99m }
    let foolsGold = { Name = "Fool's Gold"; Price = 14.98m }

    let orders = [
        { OrderId = "O1"; 
            Lines = [{ Product = rubberChicken; Count = 18 };
                         { Product = pulley; Count = 20 }]};
        { OrderId = "O2"; 
            Lines = [{ Product = fairyDust; Count = 80 }]};
        { OrderId = "O3"; 
            Lines = [{ Product = foolsGold; Count = 33 };
                         { Product = fairyDust; Count = 33 }]};
        { OrderId = "O4"; 
            Lines = [{ Product = pulley; Count = 500 }]};
        { OrderId = "O5"; 
            Lines = [{ Product = rubberChicken; Count = 18 };
                         { Product = pulley; Count = 20 }]};
        { OrderId = "O6"; 
            Lines = [{ Product = foolsGold; Count = 100 };
                         { Product = fairyDust; Count = 100 };
                         { Product = pulley; Count = 100 };
                         { Product = rubberChicken; Count = 100 }]};
        { OrderId = "O7"; 
            Lines = [{ Product = fairyDust; Count = 160 }]};
        { OrderId = "O8"; 
            Lines = [{ Product = rubberChicken; Count = 18 };
                         { Product = pulley; Count = 20 }]};
        { OrderId = "O9"; 
            Lines = [{ Product = foolsGold; Count = 260 }]};
        { OrderId = "O10"; 
            Lines = [{ Product = pulley; Count = 80 }]};
                ]

    let rec filterList f l = 
        match l with 
        | [] -> l
        | x :: xs -> (if f x then [x] else []) @ (filterList f xs)

    printfn "%A" (filterList (fun x -> x < 10) [1;3;17;20])

    let rec mapList f l = 
        match l with 
        | [] -> []
        | x :: xs -> f x :: (mapList f xs)

    printfn "%A" (mapList (fun x -> x * x) [1;5;10])

    let rec foldList f s l = 
        match l with 
        | [] -> s
        | x :: xs -> foldList f (f s x) xs 

    printfn "%A" (foldList (+) 0 [1;3;8])

    let highValueOrders orders minValue = 
        let linePrice l = decimal(l.Count) * l.Product.Price
        let orderPrice o = o.Lines |> mapList linePrice |> foldList (+) 0m

        orders |> 
        mapList (fun o -> o.OrderId, orderPrice o) |>
        filterList (fun (_, price) -> price > minValue)

    printfn "%A" (highValueOrders orders 250m)

    let highValueOrders' orders minValue = 
        let linePrice l = decimal(l.Count) * l.Product.Price
        let orderPrice o = o.Lines |> List.map linePrice |> List.fold (+) 0m

        orders |> 
        List.map (fun o -> o.OrderId, orderPrice o) |>
        List.filter (fun (_, price) -> price > minValue)

    printfn "%A" (highValueOrders' orders 250m)

    let filterSequence f s = 
        seq {
            for i in s do
                if f i then yield i
                }

    let highValueOrders'' orders minValue = 
        let linePrice l = decimal(l.Count) * l.Product.Price
        let orderPrice o = o.Lines |> Seq.map linePrice |> Seq.fold (+) 0m

        orders |> 
        Seq.map (fun o -> o.OrderId, orderPrice o) |>
        Seq.filter (fun (_, price) -> price > minValue)

    printfn "%A" (highValueOrders'' orders 250m)

module DeclaringFunctionsAndLambdaExpressions =
    let add x y = x + y

    let add' = fun x y -> x + y

    let checkThis (item: 'c) (f: 'c -> bool) : unit =
        if f item then
            printfn "HIT"
        else
            printfn "MISS"

    checkThis 5 (fun x -> x > 3)
    checkThis "hi there" (fun x -> x.Length > 5)

module DiscriminatedUnions =
    type MyEnum = 
        | First = 0
        | Second = 1

    type Product = 
        | OwnProduct of string
        | SupplierReference of int

    let p1 = Product.OwnProduct("bread")
    let p2 = SupplierReference(53)

    type Count = int

    type StockBooking = 
        | Incoming of Product * Count
        | Outgoing of Product * Count

    let bookings = 
        [
            Incoming(OwnProduct("Rubber Chicken"), 50);
            Incoming(SupplierReference(112), 18);
            Outgoing(OwnProduct("Pulley"), 6)
            Outgoing(SupplierReference(37), 40);
        ]

    type System.Int32 with
        member x.IsZero = x = 0

    let i = 5
    printfn "%A" i.IsZero

    let booking = Incoming(SupplierReference(63), 20)

    //printfn "%A" (booking.IsIncomingBooking())

    type StockBooking with
        member x.IsIncomingBooking() = 
            match x with 
            | Incoming(_,_) -> true
            | _ -> false

    printfn "%A" (booking.IsIncomingBooking())

    type 'a List = E | L of 'a * 'a List

    let ints = L(10, L(12, L(15, E)))

    printfn "%A" ints 

    let rec listSum = function
        | E -> 0
        | L(x, xs) -> x + (listSum xs)

    printfn "%d" (listSum ints)

module Exceptions =
    exception MyCustomException of int * string
        with
            override x.Message = 
                let (MyCustomException(i, s)) = upcast x
                sprintf "Int: %d Str: %s" i s

    // raise (MyCustomException(10, "Error!"))

    // failwith "Some error has occurred"

    let rec fact x = 
        if x < 0 then invalidArg "x" "Value must be >= 0"
        match x with 
        | 0 -> 1
        | _ -> x * (fact (x - 1))

    let output (o: obj) = 
        try
            let os = o :?> string
            printfn "Object is %s" os
        with 
        | :? System.InvalidCastException as ex -> printfn "Can't cast, message was: %s" ex.Message

    output 35

    let result = 
        try
            Some(10 / 0)
        with
        | :? System.DivideByZeroException -> None

    printfn "%A" result

    try
        raise (MyCustomException(3, "text"))
    with
    | MyCustomException(i, s) -> printfn "Caught custom exception with %d, %s" i s

    let getValue() = 
        try
            printfn "Returning Value"
            42
        finally
            printfn "In the finally block now"

module HelloWorldWinForms =
    open System.Windows.Forms

    type Person = { Name: string; Age: int }

    let testData = 
        [| 
            { Name = "Harry"; Age = 37 };
            { Name = "July"; Age = 41 }
        |]

    let form = new Form(Text = "F# Windows Form")

    let dataGrid = new DataGridView(Dock=DockStyle.Fill, DataSource = testData)
    form.Controls.Add(dataGrid)

    Application.Run(form)

module IDisposableHandling =
    let createDisposable f = 
        { new System.IDisposable with member x.Dispose() = f() }

    let outerFunction() = 
        use disposable = createDisposable (fun () -> printfn "Now disposing of myself")
        printfn "In outer function"

    outerFunction()

    let outerFunction'() = 
        using (createDisposable (fun () -> printfn "Now disposing of myself"))
            (fun d -> printfn "Acting on the disposable object now")
        printfn "In outer function '"

    outerFunction'()

module Inheritance =
    type CarType = 
        | Tricar = 0
        | StandardFourWheeler = 1
        | HeavyLoadCarrier = 2
        | ReallyLargeTruck = 3
        | CrazyHugeMythicalMonster = 4
        | WeirdContraption = 5

    //[<AbstractClass>]
    type Car(color: string, wheelCount: int) =
        do 
            if wheelCount < 3 then 
                failwith "We'll assume that cars must have three wheels at least"
            if wheelCount > 99 then 
                failwith "That's ridiculous"

        let carType = 
            match wheelCount with 
            | 3 -> CarType.Tricar
            | 4 -> CarType.StandardFourWheeler
            | 6 -> CarType.HeavyLoadCarrier
            | x when x % 2 = 1 -> CarType.WeirdContraption
            | _ -> CarType.CrazyHugeMythicalMonster

        let mutable passengerCount = 0

        new() = Car("red", 4)

        member x.Move() = printfn "The %s car (%A) is moving" color carType
        member x.CarType = carType

        abstract PassengerCount: int with get, set

        default x.PassengerCount with get() = passengerCount and set v = passengerCount <- v

    type Red18Wheeler() = 
        inherit Car("red", 18)

        override x.PassengerCount
            with set v = 
                if v > 2 then failwith "only two passengers allowed"
                    else base.PassengerCount <- v

    let car = Car()

    car.Move()

    let greenCar = Car("green", 4)
    greenCar.Move()

    printfn "green car has type %A" greenCar.CarType

    printfn "Car has %d passengers on board" car.PassengerCount

    car.PassengerCount <- 2
    printfn "Car has %d passengers on board" car.PassengerCount

    let truck = Red18Wheeler()
    truck.PassengerCount <- 1
    truck.PassengerCount <- 3

    let truckObject = truck :> obj
    let truckCar = truck :> Car

    let truckObjectBackToCar = truckObject :?> Car

module Interfaces =
    type IMyInterface = 
        abstract member Value: int with get

    type IDerivedInterface = 
        inherit IMyInterface

        abstract member Add: int -> int -> int 

    type MyClass() = 
        interface IMyInterface with
            member x.Value with get() = 13

    type MyOtherClass() =
        member this.Add x y = x + y

        interface IDerivedInterface with
            member i.Add x y = i.Add x y
            member x.Value = 42

    let moc = MyOtherClass()

    printfn "%A" (moc.Add 10 20)
    printfn "%A" ((moc :> IMyInterface).Value)
    printfn "%A" ((moc :> IDerivedInterface).Add 10 20)

module ListAndSequenceComprehensions =
    let output x = printfn "%A" x

    let ints = [7..13]

    output ints

    let oddValues = [1..2..20]

    output oddValues

    let values step max = [1..step..max]

    output (values 2 20)

    let ints' = seq { 7..13 }

    output ints'

    output [ for x in 7..13 -> x, x*x ]

    output [ for x in 7..13 -> 
                            printfn "Return new value now"
                            x, x*x ]
    let yieldedValues = 
        seq {
            yield 3;
            yield 42;
            for i in 1..3 do
                yield i
            yield! [5;7;8]
        }

    output (List.ofSeq yieldedValues)

    let yieldedStrings =
        seq {
            yield "this"
            yield "that"
            }

    output yieldedStrings

module List = 
    let empty = []

    let intList = [12;1;15;27]

    printfn "%A" intList

    let addItem xs x = x :: xs

    let newIntList = addItem intList 42

    printfn "%A" newIntList

    printfn "%A" (["hi"; "there"] @ ["how";"are";"you"])

    printfn "%A" newIntList.Head
    printfn "%A" newIntList.Tail
    printfn "%A" newIntList.Tail.Tail.Head

    for i in newIntList do
        printfn "%A" i

    let rec listLength (l: 'a list) = 
        if l.IsEmpty then 0
            else 1 + (listLength l.Tail)

    printfn "%d" (listLength newIntList)

    let rec listLength' l = 
        match l with 
        | [] -> 0
        | _ :: xs -> 1 + (listLength' xs)

    printfn "%d" (listLength' newIntList)

    let rec listLength'' = function
        | [] -> 0
        | _ :: xs -> 1 + (listLength' xs)

    let rec takeFromList n l =
        match n, l with 
        | 0, _ -> []
        | _, [] -> []
        | _, (x :: xs) -> x :: (takeFromList (n - 1) xs)

    printfn "%A" (takeFromList 2 newIntList)

module Modularization =
    module Adder = 
        let square x = x * x

        let add x y = x + y

    module Multiplier =
        let mult x y = x * y

    printfn "add 5 and 3 is %d" (Adder.add 5 3)

module MutableProperties =
    type CarType = 
        | Tricar = 0
        | StandardFourWheeler = 1
        | HeavyLoadCarrier = 2
        | ReallyLargeTruck = 3
        | CrazyHugeMythicalMonster = 4
        | WeirdContraption = 5


    type Car(color: string, wheelCount: int) =
        do 
            if wheelCount < 3 then 
                failwith "We'll assume that cars must have three wheels at least"
            if wheelCount > 99 then 
                failwith "That's ridiculous"

        let carType = 
            match wheelCount with 
            | 3 -> CarType.Tricar
            | 4 -> CarType.StandardFourWheeler
            | 6 -> CarType.HeavyLoadCarrier
            | x when x % 2 = 1 -> CarType.WeirdContraption
            | _ -> CarType.CrazyHugeMythicalMonster

        let mutable passengerCount = 0

        new() = Car("red", 4)

        member x.Move() = printfn "The %s car (%A) is moving" color carType
        member x.CarType = carType

        member x.PassengerCount with get() = passengerCount and set v = passengerCount <- v

    let car = Car()

    car.Move()

    let greenCar = Car("green", 4)
    greenCar.Move()

    printfn "green car has type %A" greenCar.CarType

    printfn "Car has %d passengers on board" car.PassengerCount

    car.PassengerCount <- 2
    printfn "Car has %d passengers on board" car.PassengerCount

module ObjectExpressions =
    let hiObject = { new obj() with member x.ToString() = "Hi!" }

    printfn "%A" hiObject 

    type IDeepThought = 
        abstract member TheAnswer: int with get
        abstract member AnswerString: unit -> string

    type DeepThought() = 
        interface IDeepThought with
            member x.TheAnswer = 42
            member x.AnswerString() = sprintf "The Answer is %d" (x :> IDeepThought).TheAnswer

    let htmlDeepThought = 
        let deepThought = DeepThought() :> IDeepThought
        { new IDeepThought with
            member x.TheAnswer = deepThought.TheAnswer
            member x.AnswerString() = sprintf "<b>%s</b>" (deepThought.AnswerString()) }

    printfn "%A" (htmlDeepThought.AnswerString())

    let confusedDeepThought answer = 
        { new IDeepThought with
            member x.TheAnswer = answer
            member x.AnswerString() = "uh..." }

    let cdt = confusedDeepThought 35
    printfn "%A" cdt.TheAnswer
    printfn "%A" (cdt.AnswerString())

module PartialApplication =
    let add x y = x + y
    let mult x y = x * y
    let square x = x * x

    let add' = fun x y -> x + y
    // (int -> int -> int)
    let add'' x = fun y -> x + y
    // int -> (int -> int)

    let add10'' = add'' 10
    printfn "%d" (add10'' 32)

    let add10 = add 10
    printfn "%d" (add10 32)

    let addmultipleValues a b c d = a + b + c + d

    let result = addmultipleValues 10

    let result2 = result 15

    //printfn "square 5 is %s" (square 5)

    let add''' x y =
        let result =
            x + y
        result

    let add5and3 = add 5 3

    let resultX = add (square 12) 7

module Precomputation =
    open System.Collections.Generic

    let isInList (list: 'a list) v = 
        let lookupTable = new HashSet<'a>(list)
        printfn "Lookup table created, looking up value"
        lookupTable.Contains v
    
    printfn "%b" (isInList ["hi"; "there"; "oliver"] "there")
    printfn "%b" (isInList ["hi"; "there"; "oliver"] "anna")

    let isInListClever = isInList ["hi"; "there"; "oliver"]
    printfn "%b" (isInListClever "there")
    printfn "%b" (isInListClever "anna")


    let constructLookup (list: 'a list) = 
        let lookupTable = new HashSet<'a>(list)
        printfn "Lookup table created"
        fun v -> 
            printfn "Performing lookup"
            lookupTable.Contains v

    let isInListClever' = constructLookup ["hi"; "there"; "oliver"]
    printfn "%b" (isInListClever' "there")
    printfn "%b" (isInListClever' "anna")

module RecordType = 
    type Rectangle = 
        { Width: float; Height: float }

    let rect1 = { Width = 5.3; Height = 3.4 }

    type Circle = 
        { mutable Radius : float }

        member x.RadiusSquare with get() = x.Radius * x.Radius
        member x.CalcArea() = System.Math.PI * x.RadiusSquare

    let c1 = {Radius = 3.3}

    c1.Radius <- 5.4

    type Ellipse = 
        { RadiusX: float; RadiusY: float }
        member x.GrowX dx = { x with RadiusX = x.RadiusX + dx }
        member x.GrowY dy = { x with RadiusY = x.RadiusY + dy }

    let zeroCircle = function
        | { Radius = 0.0 } -> true
        | _ -> false

    printfn "%A" (zeroCircle c1) 

    let isSquare = function 
        | { Width = width; Height = height } when width = height -> true
        | _ -> false

module Recursion = 
    open System
    open System.Threading

    let rec fact x = 
        if x = 1 then 1
        else x * (fact (x - 1))

    printfn "%d" (fact 5)

    let rec fnB() = fnA()
    and fnA() = fnB()

    let showValues() = 
        let r = Random()
        let rec dl() = 
            printfn "%d" (r.Next())
            Thread.Sleep(1000)
            dl()

        dl()

    showValues()

module SimpleClassesAndEnum = 
    type CarType = 
        | Tricar = 0
        | StandardFourWheeler = 1
        | HeavyLoadCarrier = 2
        | ReallyLargeTruck = 3
        | CrazyHugeMythicalMonster = 4
        | WeirdContraption = 5


    type Car(color: string, wheelCount: int) =
        do 
            if wheelCount < 3 then 
                failwith "We'll assume that cars must have three wheels at least"
            if wheelCount > 99 then 
                failwith "That's ridiculous"

        let carType = 
            match wheelCount with 
            | 3 -> CarType.Tricar
            | 4 -> CarType.StandardFourWheeler
            | 6 -> CarType.HeavyLoadCarrier
            | x when x % 2 = 1 -> CarType.WeirdContraption
            | _ -> CarType.CrazyHugeMythicalMonster

        new() = Car("red", 4)

        member x.Move() = printfn "The %s car (%A) is moving" color carType
        member x.CarType = carType

    let car = Car()

    car.Move()

    let greenCar = Car("green", 4)
    greenCar.Move()

    printfn "green car has type %A" greenCar.CarType

module Tuples = 
    let t1 = 12, 5, 7
    let v1, v2, _ = t1

    let t2 = "hi", true

    printfn "%A" (fst t2)
    printfn "%A" (snd t2) 

    let third t = 
        let _, _, r = t
        r

    printfn "%A" (third t1)

    let third' (_,_,r) = r

    printfn "%A" (third' t1)
