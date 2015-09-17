namespace Easj360FSharp

open System
open System.Linq
open System.Windows.Forms
open System.Windows.Controls
open System
open System.Text.RegularExpressions

module ListApp =
    
    let prepend item list = item::list

    let rec append item list =
            match list with
            | []    -> [item]
            | x::xs -> x::(append item xs)
    
    let dayInt = int DateTime.Now.DayOfWeek
    let (dayEnum: DayOfWeek) = enum dayInt

    type myDelegate = delegate of int -> unit
    let del = new myDelegate( (fun x -> printfn "%d" x ))

    //    let funWriter txt = use (System.IO.File.OpenWrite(@"textFile.txt"))
    //                                (fun sw -> sw.WriteLine(txt))

    // Events
    let eventStr1 = new Event<string>()
    eventStr1.Publish.Add(fun x -> printfn "%s" x)
    eventStr1.Trigger "Ciao"
   
    // Event Filter
    let eventStr2 = new Event<string>()    
    let newEventStr = eventStr2.Publish |> Event.filter (fun x -> x.StartsWith("H"))
    newEventStr.Add(fun x -> printfn "%s : new Event" x)
    eventStr2.Trigger "Ciao"
    eventStr2.Trigger "Hello"

       // event Partition
    let eventPartion = new Event<string>()
    let even, odd = eventPartion.Publish |> Event.partition(fun x ->    let num = Int32.Parse(x)
                                                                        let res = num % 2 = 0
                                                                        res)
//    let x = Event.partition
//               even.Add(fun x -> printfn "%s is even" x)
//               odd.Add(fun x -> printfn "%s is odd" x) 
//               eventPartion.Trigger "3"
//               eventPartion.Trigger "4"

       // Event Map
    let eventMap = new Event<string>()
    let newEventMap = eventMap.Publish |> Event.map (fun x -> "Mapped fun : " + x)
    newEventMap.Add(fun x -> printfn "%s" x)
    eventMap.Trigger "Ciao from Map Event"


        // Concat Lits
    let arrConcat = [[2; 3; 4]; [8; 7; 6; 5]; [0; 1]]

    let rec concatList l =
            if List.isEmpty l then
                []
            else
                let head = List.head l in
                let tail = List.tail l in
                head @ (concatList tail)

    let rec concatListMatch l =
            match l with 
            | head :: tail -> head @ ( concatListMatch tail)
            | [] -> []
        

    let rec fibonacci n = 
            match n with
            | 1 -> 1
            | 2 -> 2
            | x -> fibonacci (x - 1) + fibonacci (x - 2)

    type Tree<'a> =
        | Node of Tree<'a> * Tree<'a>
        | Value of 'a

    let tree = Node(    Node( Value 2, Value 3),
                            Node( Value 5, Value 8) )
    
    type InitTree<'a> =
            | Leaf of 'a
            | Node of InitTree<'a> * InitTree<'a>

    let (+) (a:int, b:int) = a + b

    let rec sumTree tree cont = //oo(x, y) =
            match tree with
            | Leaf(num)     -> cont(num)
            | Node(left, right) ->  sumTree left (fun leftCont ->
                                        sumTree right (fun rightCont ->
                                                cont( (*) leftCont rightCont ) ))
                                                //do cont leftCont
                                                //do cont rightCont))

    let tree2 = Node(Node(Node(Leaf(5), Leaf(6)), Leaf(3)), Node(Leaf(9), Leaf(7)))

    sumTree tree2 (fun x -> printfn "%d" x)

   // let setting = ConfigurationManager.AppSettings.Item("MySetting")

    let (|Regex|_|) regexPattern input =
        let regex = new Regex(regexPattern)
        let regexMatch = regex.Match(input)
        if regexMatch.Success then
            Some regexMatch.Value
        else
            None

    let (|Bool|Int|Float|) input =
        let success, res = Boolean.TryParse input
        if success then Bool(res)
        else
            let success, res = Int32.TryParse input
            if success then Int(res)
            else
                let success, res = Double.TryParse input
                if success then Float(res)
                else
                    failwith "Un match"

    
    let printChoice input =
        match input with
        | Bool b -> printfn "%b" b


    let f = ("This is a string that is becoming and Object" :> obj)

    let isThisString (s:obj) = match s with 
                        | :? string -> true
                        | _ -> false
        
    let myControls = [| (new Button() :> Control);
                        (new Button() :> Control);
                        (new Button() :> Control)
                        |]
    let uc (c: #Control) = c :> Control


    // define easier access to LINQ methods
    let select f s = Enumerable.Select(s, new Func<_,_>(f))
    let where f s = Enumerable.Where(s, new Func<_,_>(f))
    let groupBy f s = Enumerable.GroupBy(s, new Func<_,_>(f))
    let orderBy f s = Enumerable.OrderBy(s, new Func<_,_>(f))
    let count s = Enumerable.Count(s)

    
    // query string methods using functions
    let namesByFunction =
        (typeof<string>).GetMethods()
        |> where (fun m -> not m.IsStatic)
        |> groupBy (fun m -> m.Name)
        |> select (fun m -> m.Key, count m)
        |> orderBy (fun (_, m) -> m)

    // print out the data we've retrieved from about the string class    
    namesByFunction
    |> Seq.iter (fun (name, count) -> printfn "%s - %i" name count)


 
    let people = [|
            ("Ted", "Neward", 38, "Redmond", "WA")
            ("Katie", "Ellison", 30, "Seattle", "WA")
            ("Mark", "Richards", 45, "Boston", "MA")
            ("Rachel", "Reese", 27, "Phoenix", "AZ")
            ("Ken", "Sipe", 43, "St Louis", "MO")
            ("Naomi", "Wilson", 35, "Seattle", "WA")
        |]

    let names = ["Sally"; "Donny"; "Johnny"; "Josephine"; "Jose"]
    let friends = ["Sally"; "Donny"; "Jay"; "Josephine"]
    let birthdays = ["August 20th"; "April 10th"; "December 31st"; "October 3rd"]
    let places = ["Hartford, CT"; "Los Angeles, CA"; "Tokyo, Japan"; "Munich, Germany"]
    let users = ["Sally"; "Donny"; "Johnny"; "Josephine"; "Jose"]
    let lastnames = ["Struthers"; "Osmond"; "Depp"; "de Beauharnais"; "Canseco"]
    let partiallyParsedNames = ["Thomas; Richard"; "Derk; Kant; Kafka"; "Captain Crunch; Mister Rogers"]
    let strings = ["The"; "quick"; "brown"; "fox"; "jumps"; "over"; "the"; "lazy"; "dog"]
    let punctuation = [" "; " "; " "; " "; " "; " "; " "; " "; "."]

//    //filter                                                     
//    let joNames = 
//        List.filter 
//            (fun (name: string) -> "Jo" = name.Substring(0,2) ) 
//            names
//
//    //partition
//    let numbers = [1 .. 10]
//    let even1, odd1 = List.partition (fun x -> x % 2 = 0) numbers
//
//    //map
//    let lowercaseFriends = 
//        List.map 
//            (fun (str: string) -> str.ToLower()) 
//            friends
//
//    //map2
//    let friendsWithBirthdays = 
//        List.map2 
//            (fun name birthday -> sprintf "%s was born on %s" name birthday) 
//            friends birthdays
//
//    //map3
//    let friendsBirthdaysAndLocation = 
//        List.map3 
//            (fun name birthday loc -> 
//                sprintf "%s was born on %s and lives in %s" name birthday loc) 
//            friends birthdays places
//
//    //mapi
//    let usersWithUniqueNumber = List.mapi (fun i user -> (i, user)) users 
//
//    //mapi2            
//    let usersWithLastnamesAndUniqueNumbers = 
//        List.mapi2 
//            (fun i first last -> (i, sprintf "%s %s" first last)) 
//            users lastnames
//
//    //choose    
//    let lowercaseShortNames = 
//        List.choose 
//            (fun (x: string) -> 
//                match x with
//                | x when x.Length > 5 -> None
//                | x -> Some(x.ToLower()))
//            friends
//
//    //collect
//    let parsedNames = 
//        List.collect 
//            (fun (field: string) -> 
//                Array.toList (field.Split( [|"; "|], StringSplitOptions.None ))) 
//            partiallyParsedNames
//
//    //collect as choose
//    let lowercaseShortNames = 
//        List.collect 
//            (fun (x: string) -> 
//                match x with
//                | x when x.Length > 5 -> []
//                | x -> [ x.ToLower() ])
//            friends
//        
//    //reduce 
//    let longestName = 
//        List.reduce 
//            (fun (longest: string) (this:string) -> 
//                if longest.Length >= this.Length then longest else this) 
//            friends
//
//    //reduceBack
//    let longestName2 = 
//        List.reduceBack 
//            (fun (this: string) (longest:string) ->
//                if longest.Length >= this.Length then longest else this) 
//            friends
//
//    //fold
//    let totalLength = 
//        List.fold 
//            (fun acc (str: string) -> acc + str.Length)
//            0
//            strings
//   
//    //foldBack
//    let spacedStrings = 
//        List.foldBack
//            (fun str acc -> 
//                if List.isEmpty acc then [str] @ acc
//                else [str] @ [" "] @ acc)
//            strings
//            []
//        
//    //fold2
//    let totalLengthWithPunctuation =
//        List.fold2 
//            (fun acc (str: string) (punc:string) -> 
//                acc + str.Length + punc.Length)
//            0
//            strings
//            punctuation
//        
//        
//    //scan
//    let wordOffset = 
//        List.scan 
//            (fun acc (str: string) -> acc + str.Length)
//            0
//            strings
//        
//    //scanBack
//    let backwardWordOffset = 
//        List.scanBack (fun (str: string) acc -> acc + (str.Length))
//            strings
//            0
