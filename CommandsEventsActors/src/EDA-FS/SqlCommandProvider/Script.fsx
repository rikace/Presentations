#light

open System

type Command =
    | AddProduct 
    | RemoveProduct 
//    with  
//        member x.ToCommandHandler() = 
//            match x with
//            | AddProduct -> AddProductHandler 
//            | RemoveProduct -> RemoveProductHanlder
and CommandHandler =
    | AddProductHandler of string
    | RemoveProductHanlder of int
    with 
        member x.ToCommand() =
            match x with
            | AddProductHandler(s) -> AddProduct
            | RemoveProductHanlder(d) -> RemoveProduct
       

type Bus() =
    let actions = System.Collections.Generic.Dictionary<_,System.Collections.Generic.List<_>>()
    member x.Subscribe (c:Command) (f:CommandHandler -> unit) =
        let ok, k = actions.TryGetValue(c)
        match ok with
        | true -> k.Add(f)
        | false ->  let handlers = new System.Collections.Generic.List<_>()
                    handlers.Add(f)
                    actions.[c] <- handlers

    member x.Publish (c:CommandHandler) =
        let commandHanlder = c.ToCommand()
        let ok, h = actions.TryGetValue(commandHanlder)
        if ok = true then
            h |> Seq.iter(fun f -> f(c))

open System

      

[<EntryPoint>]
let main args = 
    
    let handlerAdd (a:CommandHandler) =
        match a with
        | AddProductHandler(s) -> printfn "%s ??" s
        | RemoveProductHanlder(d) -> printfn "%d number" d

    let handlerAdd2 (a:CommandHandler) =
        match a with
        | AddProductHandler(s) -> printfn "%s ?? second" s
        | RemoveProductHanlder(d) -> printfn "%d number times 2" (d * 2)

    let b = Bus()
    b.Subscribe AddProduct handlerAdd
    b.Subscribe AddProduct handlerAdd2
    b.Subscribe RemoveProduct handlerAdd
    b.Subscribe RemoveProduct handlerAdd2

    b.Publish (AddProductHandler("ciao"))
    b.Publish(RemoveProductHanlder(9))

///////////////
// Type Inference
let ``print a number`` a =
    printfn "this is a number %s" a
    
// OOP Support
[<Interface>]
type IDrinker =
    abstract Drink : unit -> unit

type Drinker =
    interface IDrinker with
        member x.Drink() = printfn "I am drinking"

type Person(name) =
    member x.Name = name        
    abstract Greet : unit -> unit
    default x.Greet() = printfn "Hi, I'm %s" x.Name    
    
type Student(name, studentID : int) =
    inherit Person(name)

    member x.StudentID = studentID

type Teacher(name, teacherId : int, drinker:IDrinker) =
    inherit Person(name)
    override x.Greet() = printfn "Hi, I'm a Teacher %s" x.Name    

    member x.TeacherID = teacherId
    member x.WhatAmIDoing() = drinker.Drink()

// Avoid Nulls
let f a = 
    a; ()

let checkMyName s = 
    match s with
    | "Riccardo" -> Some(s)
    | _ -> None

// Record Type
(*Records represent simple aggregates of named values, optionally with members.*)
type Person' = {FirstName:string; LastName:string; Age:int }
                with override x.ToString() = sprintf "Full Name %s %s" x.FirstName x.LastName

let me0Years = {FirstName="Ricky"; LastName="Terrell"; Age=0}
let me = {me0Years with Age=37}
let meFullName = {me with FirstName="Riccardo"}

let checkMe record = 
    match record with
    | {FirstName="Riccardo"} -> printfn "This is me"
    | _ -> printfn "Wrong Person"

me = meFullName

(*Discriminated unions are used a lot in F# code.  A discriminated union is a data type with a finite number of distinct alternative representations.  If you have used "union" in C/C++, you can think of discriminated unions as a somewhat similar construct; the main differences are that F# discriminated unions are type-safe (every instance knows which alternative is 'active') and that F# discriminated unions work with pattern-matching *)

type ADiscriminatedUnionType =
    | Alternative1 of int * string
    | Alternative2 of bool
    | Alternative3 

// constructing instances 
let du1 = Alternative1(42,"life") 
let du3 = Alternative3 

let f' du = // pattern matching 
    match du with
    | Alternative1(x,s) -> printfn "x is %d, s is %s" x s
    | Alternative2(b) -> printfn "b is %A" b
    | Alternative3 -> printfn "three"

f' du1  // x is 42, s is life 
f' du3 // three

// Pipe Operator
let fun' (x:int):string = x.ToString()
let fun'' (s:string) = int(s) * 2
let fun''' x = x / 2 + 1
let fun'''' = fun' >> fun'' >> fun'''
let resutl = fun'''' 2

[1..10] |> List.filter(fun f -> f % 2 = 0) |> List.map(fun f -> f * f) |> List.reduce (+)  
[1..10] |> List.filter(fun f -> f % 2 = 0) |> List.map(fun f -> f * f) |> List.fold (+) 0

// Object Expression
(*An object expression is an expression that creates a new instance of a dynamically created, anonymous object type that is based on an existing base type, interface, or set of interfaces.*)
let people = new System.Collections.Generic.List<_>(
                [|
                    { FirstName = "Riccardo"; LastName = "Terrell"; Age=37 }
                    { FirstName = "Bugghina"; LastName = "Terrell"; Age=6 }
                |])
let peopleSort = { new System.Collections.Generic.IComparer<Person'> with
                        member this.Compare(l, r) =
                            if l.FirstName = r.FirstName then 1
                            elif l.LastName = r.LastName then 0
                            else -1   }      
people.Sort(peopleSort) 
people |> Seq.toList |> List.iter (printfn "%A")

// Active Patterns
(*Active patterns enable you to define named partitions that subdivide input data, so that you can use these names in a pattern matching expression just as you would for a discriminated union. You can use active patterns to decompose data in a customized manner for each partition*)
let (|ToInt|_|) value =
    let success, result = System.Int32.TryParse(value)
    if success then Some(result)
    else None

let (|ToBool|_|) value =
    let success, result = System.Boolean.TryParse(value)
    if success then Some(result)
    else None

let testPartialActivePattern value =
    match value with
    | ToInt i -> printfn "Value is Int %d" i
    | ToBool b -> printfn "Value is bool %b" b
    | _ -> printfn "Problem"  

testPartialActivePattern "3"
testPartialActivePattern "true"

let (|Even|Odd|) value = if value % 2 = 0 then Even else Odd

let TestNumber value = 
    match value with
    | Even -> printfn "%d is Even" value
    | Odd -> printfn "%d is Odd" value

TestNumber 7
TestNumber 8