#light

// This file is a script that can be executed with the F# Interactive.  
#load "ParallelEx.fs"
#load "ArrayParallelSort.fs"

open System.Net
open System.IO
open System.Collections.Generic
open System
open Easj360.Parallel

// Sort which runs a serial
let items0 = [| 9; 7; 5; 3; 1 |]
items0  
|> Array.Parallel.Merge.sortInPlace   

printfn "Serial: %A" items0

// Sort simple collection of numbers
let items1 = [| 10000 .. -1 .. 1 |]

items1
|> Array.Parallel.Merge.sort
|> printfn "Simple New: %A"

items1
|> Array.Parallel.Merge.sortInPlace   

printfn "Simple In Place: %A" items1

// Base parallel sort test
let items2 = [| for f in 0.0 .. 0.1 .. 100.0 -> sin f |]
items2
|> Array.Parallel.Merge.sortInPlace   

printfn "Sorted: %A" items2

// Parallel sort with a projection
let items3 = [| for f in 0.0 .. 0.1 .. 100.0 -> sin f |]
items3
|> Array.Parallel.Merge.sortInPlaceBy (fun item -> abs item)

printfn "Sorted ABS: %A" items3


// Some 5 million item array performance testing
#load "ParallelEx.fs"

open System
open MSDN.FSharp.Parallel

let rnd = System.Random()

let recordTime func =
    GC.Collect()
    GC.Collect(GC.MaxGeneration)
    GC.WaitForFullGCComplete() |> ignore
    GC.WaitForPendingFinalizers()
    let sw = new System.Diagnostics.Stopwatch()
    sw.Start()
    func()
    sw.Elapsed


let writeTime (message:string) (sortCount:int) (timespan : TimeSpan) = 
    printfn "%s: Sort took %f seconds : Element count = %i" message timespan.TotalSeconds sortCount

let itemsBase = [| for f in 0 .. 1 .. 2500000 -> (rnd.NextDouble() - 0.8) * 1000.0 |]

// Base sort
let items7 = Array.copy itemsBase
let items8 = Array.copy itemsBase
let items9 = Array.copy itemsBase

recordTime (fun () ->
    items7
    |> Array.Parallel.Quick.sortInPlace) 
|> writeTime "ParallelQuickInPlace" 5000000

recordTime (fun () ->
    ParallelMergeSort.SortInPlace(items8)) 
|> writeTime "ParallelMergeInPlace" 5000000

recordTime (fun () ->
    items9
    |> Array.sortInPlace) 
|> writeTime "SequentialInPlace" 5000000

// Base sort new array
let items7n = Array.copy itemsBase
let items8n = Array.copy itemsBase
let items9n = Array.copy itemsBase

recordTime (fun () ->
    items7n
    |> Array.Parallel.Quick.sort
    |> ignore) 
|> writeTime "ParallelQuick" 5000000

recordTime (fun () ->
    items8n
    |> Array.Parallel.Merge.sort
    |> ignore) 
|> writeTime "ParallelMerge" 5000000

recordTime (fun () ->
    items9n
    |> Array.sort
    |> ignore) 
|> writeTime "Sequential" 5000000

// With sort
let items7w = Array.copy itemsBase
let items8w = Array.copy itemsBase
let items9w = Array.copy itemsBase

recordTime (fun () ->
    items7w
    |> Array.Parallel.Quick.sortInPlaceWith compare)
|> writeTime "ParallelQuickInPlaceWith" 5000000

recordTime (fun () ->
    items8w
    |> Array.Parallel.Merge.sortInPlaceWith compare)
|> writeTime "ParallelMergeInPlaceWith" 5000000

recordTime (fun () ->
    items9w
    |> Array.sortInPlaceWith compare)
|> writeTime "SequentialInPlaceWith" 5000000

// By sort
let items6b = Array.copy itemsBase
let items7b = Array.copy itemsBase
let items8b = Array.copy itemsBase
let items9b = Array.copy itemsBase

let pcompare a b = compare ((sqrt << abs) a) ((sqrt << abs) b)
recordTime (fun () ->
    items6b
    |> Array.Parallel.Quick.sortInPlaceWith pcompare) 
|> writeTime "ParallelQuickInPlaceByWith" 5000000

recordTime (fun () ->
    items7b
    |> Array.Parallel.Quick.sortInPlaceBy (fun item -> (sqrt << abs) item)) 
|> writeTime "ParallelQuickInPlaceBy" 5000000

recordTime (fun () ->
    items8b
    |> Array.Parallel.Merge.sortInPlaceBy (fun item -> (sqrt << abs) item))
|> writeTime "ParallelMergeInPlaceBy" 5000000

recordTime (fun () ->
    items9b
    |> Array.sortInPlaceBy (fun item -> (sqrt << abs) item)) 
|> writeTime "SequentialInPlaceBy" 5000000






let url = new Uri("")
let mem = new MemoryStream()
let wc = new WebClient()
wc.DownloadDataCompleted.Add(fun e ->
    do ignore( mem.BeginWrite(e.Result, 0, e.Result.Length, (fun ar ->
                                mem.EndWrite(ar)), null)))
wc.DownloadDataAsync(url)

////////////////////////////////////////    
[<AbstractClass>]
type Shape(x:int) =
    let mutable x0 = x
    
    member this.GetX with get() = x0 and set v = x0 <- v
    
    abstract Line : string with get
    
type Rect(x:int) =
    inherit Shape(x)
    //override this.Line with get() = x.ToString()
    default this.Line with get() = x.ToString()
        
type Pris(o:Shape) =
    member x.GetData() =
        o.Line
       
let r = new Rect(9)
let p = new Pris(r)
p.GetData()


type IComp =
    interface
        abstract member GetData : unit -> int
    end
    
type TestComp() =
    interface IComp with
        member this.GetData() =
            8

type TestGetComp(x:#IComp) =
    member this.Get() =
        x.GetData()
            
let tc = new TestComp()            
let t = new TestGetComp(tc)
t.Get()

[<Sealed>]
type sealedtype() =
    member x.Data() =
     0

////////////////////////////////////////

let rec revert_list l =   
    match l with   
    | [] -> []   
    | x::rest -> (revert_list rest) @ [x]

let rec insert_after elem newelem l =
    match l with    
    | [] -> [newelem]    
    | x::rest -> if x = elem then                    
                    (x::newelem::rest)                 
                 else                     
                    x::(insert_after elem newelem rest)

let rec insert_before elem newelem l =    
    match l with    
    | [] -> [newelem]    
    | x::rest -> if x = elem then                    
                    (newelem::x::rest)                 
                 else                    
                    x::(insert_before elem newelem rest)        

let list1 = [1;2;3;4;-4;-3;-2;-1]
let list2 = insert_after 4 6 list1
let list3 = insert_before 6 5 list2

let rec remove_if l predicate =    
    match l with    
    | [] -> []    
    | x::rest ->    if predicate(x) then                    
                        (remove_if rest predicate)                 
                    else                     
                        x::(remove_if rest predicate)

let list1 = [1;2;3;4;-4;-3;-2;-1]
let list2 = remove_if list1 (fun x -> (abs x &&&1) = 1)
let list3 = remove_if list1 (fun x -> (abs x &&&1) = 0)
let list4 = remove_if list1 (fun x -> x < 0)

printfn "%a" output_any list1
printfn "%a" output_any list2
printfn "%a" output_any list3
printfn "%a" output_any list4

////////////////////////////////////////////////

// Static Upcast :>  is where an instance of a derived class is cast as one its base classes

[<AbstractClass>]
type animal() = 
    abstract member Legs : int

[<AbstractClass>]
type dog() =
    inherit animal() 
        abstract member Description : string
        default this.Description = ""
        override this.Legs = 4

type pug() =
    inherit dog()
    override this.Description = "Bugghina"

let bugghina = new pug()
let bella = bugghina :> dog
let amore = bugghina :> animal

// Dynamic Cast :?>  is when you cast a base type as derived type, or cast something down the inheritance tree

let objDog = bugghina :> obj
let objAsBugghina = objDog :?> dog

////////////////////////////////////////
let rec isOdd n = (n = 1) || isEven (n - 1)
and isEven n = (n = 0) || isOdd (n - 1)

isOdd(9)

////////////////////////////////////////
let (===) str (rgx:string) =
    System.Text.RegularExpressions.Regex.Match(str, rgx).Success

"Bugghina e' bella" === "Bugghina (.*)"

////////////////////////////////////////
let mWhen v =
    match v with
    | _ when v > 10 -> "10"
    | _ when v > 5  ->  "5"
    | _ -> "0"

mWhen 11

////////////////////////////////////////
type DogRecord = { Name: string; Legs: int; mutable Age : int}
                 member this.TellMeTheAge = printfn "%s is %d years old" this.Name this.Age

let myBugghina = { Name= "Bugghina"; Legs=4; Age = 4}
myBugghina.TellMeTheAge
myBugghina.Age <- 5

let arrRectangular : int[,] = Array2D.zeroCreate 2 2
arrRectangular.[0,0] <- 0
arrRectangular.[0,1] <- 1
arrRectangular.[1,0] <- 2
arrRectangular.[1,1] <- 3

let jaggedArr : int[][] = Array.zeroCreate 2
jaggedArr.[0] <- Array.init 2 (fun x -> x)
jaggedArr.[1] <- Array.init 4 (fun x -> x)

type StaticVer() =
    static let mutable Vertical = 0

////////////////////////////////////////
[<Struct>]
type PointStruct(x:int, y:int) =
    member this.X = x
    member this.Y = y

////////////////////////////////////////
type Point =
    val m_x : int
    val m_y :int

    new (x,y) = {m_x = x; m_y = y}

    new () = {m_x = 0; m_y = 0}

    member this.Length =
        this.m_x + this.m_y

//[<Flags>]
//[<Enum>]
type FlagEnum =
    | opt1 = 0
    | Opt2 = 1

////////////////////////////////////////
let (|FileExtension|) (filePath:string) = System.IO.Path.GetExtension(filePath)

let determinateIfFileIsImage (filePath:string) =
    match filePath with
    | FileExtension ".jpg"
    | FileExtension ".png"
    | FileExtension ".gif"
        -> "The File is image"
    | _ -> "The File is something else"
    
let (|Paragraph|Sentence|Word|WhiteSpace|) (input:string) =
    let input = input.Trim()
    if input = "" then WhiteSpace
    elif input = "." then Paragraph 10
    elif input = "ssss" then Sentence (input.Split([|" "|], StringSplitOptions.None))
    else Word(input)

let rec countLetters str =
    match str with
    | WhiteSpace -> 0
    | Word x -> x.Length
    | Sentence words -> Array.length words
    | Paragraph num -> num
    
////////////////////////////////////////
// Slow implementation...
let removeConsecutiveDupes1 lst =

    let foldFunc acc item =
        let lastLetter, dupesRemoved = acc
        match lastLetter with
        | Some(c) when c = item  
                  -> Some(c), dupesRemoved
        | Some(c) -> Some(item), dupesRemoved @ [item]
        | None    -> Some(item), [item]

    let (_, dupesRemoved) = List.fold foldFunc (None, []) lst
    dupesRemoved
    
// Fast implementation...
let removeConsecutiveDupes2 lst = 
    let f item acc =
        match acc with
        | [] 
            -> [item]
        | hd :: tl when hd <> item 
            -> item :: acc
        | _ -> acc
    
    List.foldBack f lst []

////////////////////////////////////////
type WebScraper(url) =

    let downloadWebpage (url : string) =
        let req = WebRequest.Create(url)
        let resp = req.GetResponse()
        let stream = resp.GetResponseStream()
        let reader = new StreamReader(stream)
        reader.ReadToEnd()
        
    let extractImageLinks html =
        let results = System.Text.RegularExpressions.Regex.Matches(html, "<img src=\"([^\"]*)\"")
        [
            for r in results do
                for grpIdx = 1 to r.Groups.Count - 1 do
                    yield r.Groups.[grpIdx].Value 
        ] |> List.filter (fun url -> url.StartsWith("http://"))
    
    let downloadToDisk (url : string) (filePath : string) =
        use client = new System.Net.WebClient()
        client.DownloadFile (url, filePath)

    let scrapeWebsite destPath (imageUrls : string list) =       
        imageUrls
        |> List.map(fun url -> 
                let parts = url.Split( [| '/' |] )
                url, parts.[parts.Length - 1])
        |> List.iter(fun (url, filename) -> 
                downloadToDisk url (Path.Combine(destPath, filename)))
                
    // Add class fields
    let m_html   = downloadWebpage url
    let m_images = extractImageLinks m_html
    
    // Add class members
    member this.SaveImagesToDisk(destPath) =
        scrapeWebsite destPath m_images 

///////////////////////////////////

type SetAction = Added | Removed

type SetOperationEventArgs<'a>(value : 'a, action : SetAction) =
    inherit System.EventArgs()
    
    member this.Action = action
    member this.Value = value

type SetOperationDelegate<'a> = delegate of obj * SetOperationEventArgs<'a> -> unit

// Contains a set of items that fires events whenever
// items are added.
type NoisySet<'a when 'a : comparison>() =
    let mutable m_set = Set.empty : Set<'a>
    
    let m_itemAdded = 
        new Event<SetOperationDelegate<'a>, SetOperationEventArgs<'a>>()

    let m_itemRemoved = 
        new Event<SetOperationDelegate<'a>, SetOperationEventArgs<'a>>()    
    
    member this.Add(x) = 
        m_set <- m_set.Add(x)
        // Fire the 'Add' event
        m_itemAdded.Trigger(this, new SetOperationEventArgs<_>(x, Added))
        
    member this.Remove(x) =
        m_set <- m_set.Remove(x)
        // Fire the 'Remove' event
        m_itemRemoved.Trigger(this, new SetOperationEventArgs<_>(x, Removed))

    // Publish the events so others can subscribe to them
    member this.ItemAddedEvent   = m_itemAdded.Publish
    member this.ItemRemovedEvent = m_itemRemoved.Publish

// Using events
let s = new NoisySet<int>()

let setOperationHandler =
    new SetOperationDelegate<int>(
        fun sender args ->
            printfn "%d was %A" args.Value args.Action
    )

s.ItemAddedEvent.AddHandler(setOperationHandler)
s.ItemRemovedEvent.AddHandler(setOperationHandler)

////////////////////////////////////////////////////

type ClockUpdateDelegate = delegate of int * int * int -> unit

type Clock() =

    let m_event = new DelegateEvent<ClockUpdateDelegate>()

    member this.Start() =
        printfn "Started..."
        while true do
            // Sleep one second...
            Threading.Thread.Sleep(1000)
        
            let hour   = DateTime.Now.Hour
            let minute = DateTime.Now.Minute
            let second = DateTime.Now.Second
           
            m_event.Trigger( [| box hour; box minute; box second |] )

    member this.ClockUpdate = m_event.Publish

////////////////////////////////////////
#r "System.Windows.Forms.dll"
    // reference an Assembly

#I @"C:\WINDOWS\Microsoft.NET"
    // add a new directory to the assembly F# searvh path

//#load "SomeScript.fs"
    // load and open a F# file

let rec filesUnder path =
    seq {   yield! System.IO.Directory.GetFiles(path)
            for subDir in System.IO.Directory.GetDirectories(path) do
                yield! filesUnder subDir    }

////////////////////////////////////////
let asyncTaskX = async { failwith "error" }

asyncTaskX
|> Async.Catch 
|> Async.RunSynchronously
|> function 
   | Choice1Of2 result     -> printfn "Async operation completed: %A" result
   | Choice2Of2 (ex : exn) -> printfn "Exception thrown: %s" ex.Message

////////////////////////////////////////
let cancelableTask =
    async {
        printfn "Waiting 10 seconds..."
        for i = 1 to 10 do 
            printfn "%d..." i
            do! Async.Sleep(1000)
        printfn "Finished!"
    }
   
// Callback used when the operation is canceled
let cancelHandler (ex : OperationCanceledException) = 
    printfn "The task has been canceled."

Async.TryCancelled(cancelableTask, cancelHandler)
|> Async.Start

Async.CancelDefaultToken()

////////////////////////////////////////
let superAwesomeAsyncTask = async { return 5 }

Async.StartWithContinuations(
    superAwesomeAsyncTask,
    (fun (result : int) -> printfn "Task was completed with result %d" result),
    (fun (exn : Exception) -> printfn "threw exception"),
    (fun (oce : OperationCanceledException) -> printfn "OCE"))

////////////////////////////////////////
let computation = Async.TryCancelled(cancelableTask, cancelHandler)
let cancellationSource = new System.Threading.CancellationTokenSource()

Async.Start(computation, cancellationSource.Token)

cancellationSource.Cancel()

////////////////////////////////////////

let g f s = f(s);;
//val g : ('a -> 'b) -> 'a -> 'b

let c v = Int32.TryParse(v);;
//val c : string -> bool * int

g c "3";;
//val it : bool * int = (true, 3)

//Option.map;;
//val it : (('a -> 'b) -> 'a option -> 'b option) = <fun:clo@9>

let places = [ ("A", 1), ("B", 2), ("C", 3)];;

//val places : ((string * int) * (string * int) * (string * int)) list =   [(("A", 1), ("B", 2), ("C", 3))]

 let get(p) = match p with
              | n when n > 2 -> "Selected B"
              | n when n = 2 -> "Selected C"
              | _ -> "Boooo";;

//val get : int -> string

places |> List.map (fun (_, p) -> get(p));;
let places2 = [ ("A", 1); ("B", 2); ("C", 3)];;
//val places : (string * int) list = [("A", 1); ("B", 2); ("C", 3)]

places2 |> List.map (fun (_, p) -> get(p));;
//val it : string list = ["Boooo"; "Selected C"; "Selected B"]

places2 |> List.map (fun p -> get((snd p)));;
//val it : string list = ["Boooo"; "Selected C"; "Selected B"]
places2 |> List.map snd |> List.map get;;
//val it : string list = ["Boooo"; "Selected C"; "Selected B"]

let run f = f();;
//val run : (unit -> 'a) -> 'a

run (fun _ -> printfn "ciao");;
//ciao
//val it : unit = ()


////////////////////////////////////////
// Record and cloning

type Point2D = {X: float; Y:float}
let p2d = {X=2.0; Y=3.4}
let p2dCloning = { p2d with Y=5.}

for (val1, val2) in [("Banana", 1); ("Kiwi", 2)] do printfn "fruit type %s" val1

let dics = new Dictionary<int, string>(HashIdentity.Structural)
let mutable res = ""
let foundIt = dics.TryGetValue(1, &res)
let foundIt2 value = dics.TryGetValue(1)

////////////////////////////////////////

let using (ie:#IDisposable) f =
    try f(ie)
    finally ie.Dispose()

let writeToFile () =
    using(File.CreateText("nameFilHere"))(fun outP -> outP.WriteLine("some Text Here"))

let rec map (f: 'T -> 'U) (l: 'T list) =
    match l with
    | h :: t -> f h :: map f t
    | [] -> []

let emptyList<'T> q : seq<'T list> = Seq.init q (fun _ -> [])
emptyList<int> 100
emptyList<string> 3

////////////////////////////////////////

type f (?value:string, ?num:int) =
    let value = defaultArg value ""
    let num = defaultArg num 0


let matrix rows cols = Array2D.zeroCreate<int> rows cols     

////////////////////////////////////////

[<AbstractClass>]
type TextWriter() =
    abstract Write : string -> unit


type HtmlWriter() =   
    let sink =
        { new TextWriter() with
                member x.Write s =
                    System.Console.Write s }

    member x.Write s =
        sink.Write("<tag>")
        sink.Write(s)
        sink.Write("</tag>")

////////////////////////////////////////

let radInput() =
    let s = Console.ReadLine()
    match Int32.TryParse(s) with
    | true, p -> Some(p)
    | _ -> None

let isPrime i =
    let lim = int(sqrt(float(i)))
    let rec check j = 
        j > lim || (i % j <> 0 && check (j+1))
    check 2

type System.Int32 with
    member x.IsPrime = isPrime x

////////////////////////////////////////

let rec pairwiseList l =
    match l with
    | [] | [_] -> []
    | h1::(h2::_ as t) -> (h1,h2) :: pairwiseList t


////////////////////////////////////////

let fibonacci n : bigint =  
    let rec f a b n =    
        match n with    
        | 0 -> a    
        | 1 -> b    
        | n -> (f b (a + b) (n - 1))  
    f (bigint 0) (bigint 1) n

let rec fibS = seq { yield! [0;1];                    
                    for (a,b) in Seq.zip fibS (Seq.skip 1 fibS) -> a+b}

//let fibonacci = Seq.unfold (fun (x, y) -> Some(x, (y, x + y))) (0I,1I)fibonacci |> Seq.nth 10000

let rec inline fib n = if n <= 2 then 1 else fib (n-1) + fib (n+1)

let memoize (f: 'T -> 'U) =
    let t = new Dictionary<'T, 'U>(HashIdentity.Structural)
    fun n ->
        if t.ContainsKey(n) then t.[n]
        else let res = f n
             t.Add(n, res)
             res 
             
let rec fibFast = memoize (fun n -> if n <= 2 then 1 else fibFast (n-1) + fibFast (n+1))                                   

////////////////////////////////////////

let rec sumList(lst) =
    let rec sumListUtil(lst, total) =
        match lst with
        | [] -> total
        | hd::tl -> let ntotal = hd + total
                    sumListUtil(tl, ntotal)
    sumListUtil(lst, 0)

    match l with
    | []    -> 0
    | h::t -> h + sumList(t)

let rec aggregateList (op:int -> int -> int) init list =
    match list with
    | []    -> init
    | h::t  ->  let resultRest = aggregateList op init list
                op resultRest h

let add a b = a + b
let mul a b = a * b

aggregateList add 0 [1..5]

let rec last l =
    match l with
    | [] -> []
    | [h] -> h
    | h::t -> last t

let rec mapAcc f inputList acc =
    match inputList with
    | [] -> List.rev acc
    | h::t mapAcc f t (f h :: acc)

//mapAcc (fun f -> f * f) [1;2;3;4] []
////////////////////////////////////////

let forkJoinParallel (taskSeq) =
    Async.FromContinuations(fun (cont, econt, ccont) ->
        let tasks = Seq.toArray taskSeq
        let count = ref tasks.Length
        let results = Array.zeroCreate tasks.Length
        tasks |> Array.iteri(fun i p ->
            Async.Start( async{
                let! res = p
                results.[i] <-res
                let n = System.Threading.Interlocked.Decrement(count)
                if n=0 then cont results
            })))

let readLock (rwlock:System.Threading.ReaderWriterLock) f =
    rwlock.AcquireReaderLock(Timeout.Infinite)
    try
        f()
    finally
        rwlock.ReleaseReaderLock()

let writeLock (rwlock:System.Threading.ReaderWriterLock) f =
    rwlock.AcquireWriterLock(Timeout.Infinite)
    try
        f()
    finally
        rwlock.ReleaseWriterLock()

let rw = new System.Threading.ReaderWriterLock()
readLock rw (fun f -> ())

////////////////////////////////////////

type Client = {
  Name : string
  Income : int
  YearsInJob : int
  UsesCreditCard : bool
  CriminalRecord : bool }

type QueryInfo =
  { Title : string
    // Member representing the behavior
    Test : Client -> bool
    // References to the second type
    Positive : Decision
    Negative : Decision }
    
// Make the declaration recursive using 'and' keyword 
and Decision = 
  | Result of string  
  | Query of QueryInfo // Reference to the first type


// Listing 8.13 Decision tree for testing clients

// Root node on level 1
let rec tree = 
    Query({ Title = "More than $40k" 
            Test = (fun cl -> cl.Income > 40000)
            Positive = moreThan40; Negative = lessThan40 })
// First option on the level 2
and moreThan40 = 
    Query({ Title = "Has criminal record"
            Test = (fun cl -> cl.CriminalRecord)
            Positive = Result("NO"); Negative = Result("YES") })
// Second option on the level 2
and lessThan40 = 
    Query({ Title = "Years in job"
            Test = (fun cl -> cl.YearsInJob > 1)
            Positive = Result("YES"); Negative = usesCredit })
// Additional question on level 3
and usesCredit = 
    Query({ Title = "Uses credit card"
            Test = (fun cl -> cl.UsesCreditCard)
            Positive = Result("YES"); Negative = Result("NO") })

// Recursive function declaration
let rec testClientTree(client, tree) =
  match tree with
  | Result(msg) ->
      // The case with the final result
      printfn "  OFFER A LOAN: %s" msg
  | Query(qi) ->
      // The case containing a query
      let s, case = if (qi.Test(client)) then "yes", qi.Positive
                    else "no", qi.Negative
      printfn "  - %s? %s" qi.Title s
      // Recursive call on the selected sub-tree
      testClientTree(client, case)

let john = 
  { Name = "John Doe"  
    Income = 40000
    YearsInJob = 1
    UsesCreditCard = true
    CriminalRecord = false }

// Test the code interactively
testClientTree(john, tree)

////////////////////////////////////////

// 'Rect' type with processing functions
module Listing_9_1 = 
  // Type declaration
  type Rect =
    { Left   : float32
      Top    : float32
      Width  : float32
      Height : float32 }                       

  // Shrinks the rectangle
  let deflate(rc, wspace, hspace) = 
    { Top = rc.Top + wspace
      Left = rc.Left + hspace
      Width = rc.Width - (2.0f * wspace)
      Height = rc.Height - (2.0f * hspace) }
  
  // Conversion to 'System.Drawing' representation
  let toRectangleF(rc) = 
    RectangleF(rc.Left, rc.Top, rc.Width, rc.Height);;
    
////////////////////////////////////////
// Create object implementing 'IEqualityComparer<string>'
let equality = 
  // Utility function that removes spaces from the string
  let replace(s:string) = s.Replace(" ", "")
  { new IEqualityComparer<_> with
      // Compare strings, ignoring the spaces
      member x.Equals(a, b) = 
        String.Equals(replace(a), replace(b))
      // Get hash code of the string without spaces
      member x.GetHashCode(s) = 
        replace(s).GetHashCode() }
      
// Mutate the collection
let dict = new Dictionary<_, _>(equality)
dict.Add("100", "Hundred")
dict.Add("1 000", "thousand")
dict.Add("1 000 000", "million")

// Get value using the custom comparer
dict.["100"]
dict.["10 00"]
dict.["1000000"]

////////////////////////////////////////
let changeColor clr = 
  // Store the original color
  let orig = System.Console.ForegroundColor
  // Set the new color immediately
  System.Console.ForegroundColor <- clr
  // Create 'IDisposable' value
  { new IDisposable with
      member x.Dispose() = 
        // Restore the original color inside 'Dispose' 
        System.Console.ForegroundColor <- orig }
////////////////////////////////////////

type ClientTest = 
  abstract Test : Client -> bool
  abstract Report : Client -> unit

// Implementing interface in a class

// Implicit class declaration
type CoefficientTest(income, years, min) =

  // Local helper functions
  let coeff cl =
    ((float cl.Income)*income + (float cl.YearsInJob)*years)
  let report cl =
    printfn "Coefficient %f is less than %f." (coeff cl) min

  // Standard public method of the class
  member x.PrintInfo() =
    printfn "income*%f + years*%f > %f" income years min

  // Interface implementation using helpers
  interface ClientTest with
    member x.Report cl = report cl
    member x.Test cl = (coeff cl) < min

////////////////////////////////////////
let rec addSimple(a, b) = 
    // Prints info for debugging 
    printfn "adding %d + %d" a b
    a + b

  // Addition optimized using memoization
let add = 
    // Initialize the cache 
    let cache = new Dictionary<_, _>()
    // Created function uses the private cache
    (fun x ->
        // Read the value from the cache or calculate it
        match cache.TryGetValue(x) with
        | true, v -> v
        | _ -> let v = addSimple(x)
               cache.Add(x, v)
               v)

// Calls the 'addSimple' function
add(2, 3)

// Value is obtained from the cache
add(2, 3)

// Section Reusable memoization

// Generic memoization function
let memoize(f) =    
  // Initialize cache captured by the closure
  let cache = new Dictionary<_, _>()
  (fun x ->
      let succ, v = cache.TryGetValue(x)
      if succ then v else 
        let v = f(x)
        cache.Add(x, v)
        v)

////////////////////////////////////////
// Working with lists efficiently
// Naïve list processing functions
let rec mapN f ls =
  match ls with
  | [] -> []
  | x::xs -> 
      f(x) :: (mapN f xs)

let rec filterN f ls =
  match ls with
  | [] -> []
  | x::xs -> 
      let xs = (filterN f xs)
      if f(x) then x::xs else xs

// Tail recursive list processing functions
let map f ls =
  let rec map' f ls acc =
    match ls with
    | [] -> acc
    | x::xs -> map' f xs (f(x) :: acc)
  map' f ls [] |> List.rev

let filter f ls =
  let rec filter' f ls acc =
    match ls with
    | [] -> acc
    | x::xs -> 
      filter' f xs 
        (if f(x) then x::acc else acc)        
  filter' f ls [] |> List.rev

// Testing the naive and tail-recursive version..
let large = [ 1 .. 100000 ]

// Tail recursive function works fine
large |> map (fun n -> n*n)
// Non-tail recursive function causes stack overflow
large |> mapN (fun n -> n*n)

////////////////////////////////////////
// Efficiency of list functions
// Adding elements to a list (F# interactive)

// Appends simply using cons operator
let appendFront el list = el::list

// Appends to the end using a recursive function
let rec appendEnd el list =
  match list with
  | []    -> [el] // Append to an empty list
  | x::xs -> x::(appendEnd el xs) // Recursive call append to the tail

////////////////////////////////////////
// Tree is a leaf with value or a node containing sub-trees
type IntTree =
  | Leaf of int
  | Node of IntTree * IntTree

// Recursive function calculating sum of elements
let rec sumTree(tree) =
  match tree with
  // Sum of a leaf is its value
  | Leaf(n)    -> n
  // Recursively sum values in the sub-trees
  | Node(l, r) -> sumTree(l) + sumTree(r)

let tree = 
  Node(Node(Node(Leaf(5), Leaf(8)), Leaf(2)),
       Node(Leaf(2), Leaf(9)));;
sumTree(tree)

// Deep recursion causes stack overflow!
let imbalancedTree =
  test2 |> List.fold_left(fun st v ->
      // Add node with previous tree on the right
      Node(Leaf(v), st)) (Leaf(0))
sumTree(imbalancedTree)

// --------------------------------------------------------------------------
// Writing code using continuations
// Sum elements of a tree using continuations (F# interactive)

// Return value by calling the continuation
let rec sumTreeCont tree cont =
  match tree with
  | Leaf(n)    -> cont(n)
  | Node(l, r) -> 
      // Recursively sum left sub-tree
      sumTreeCont l (fun n ->
        // Then recursively sum right sub-tree
        sumTreeCont r (fun m ->
          // Finally, call the continuation with the result
          cont(n + m)))
          
// Print the result inside the continuation
sumTreeCont imbalancedTree (fun r ->
  printfn "Result is: %d" r)

// Returning sum from the continuation
sumTreeCont imbalancedTree (fun a -> a)
////////////////////////////////////////

let rec nums = seq {    yield 1
                        for n in nums do yield n + 1 } |> Seq.cache

////////////////////////////////////////
let cities = [ ("New York", "USA"); ("London", "UK");
               ("Cambridge", "UK"); ("Cambridge", "USA") ]
let entered = [ "London"; "Cambridge" ]
////////////////////////////////////////
let rec allFiles dir =
    Seq.append  
        (dir |> System.IO.Directory.GetFiles)
        (dir |> System.IO.Directory.GetDirectories |> Seq.map allFiles |> Seq.concat)
        
let allFiles dir =
    seq { for file in System.IO.Directory.GetFiles(dir) do
            let creationTime = System.IO.File.GetCreationTime(file)
            yield (file,creationTime)
        }
////////////////////////////////////////
let xattr s (el:XElement) = 
  el.Attribute(XName.Get(s)).Value
// Returns child node with the specified name
let xelem s (el:XContainer) = 
  el.Element(XName.Get(s))
// Returns child elements with the specified name
let xelems s (el:XContainer) = 
  el.Elements(XName.Get(s))
// Returns the text inside the node
let xvalue (el:XElement) = 
  el.Value
////////////////////////////////////////

// Message passing concurrency

type Message = 
  | ModifyState of int
  | Block
  | Resume

// Mailbox processor using state machine  
let mbox = MailboxProcessor.Start(fun mbox ->
    // Represents the blocked state
    let rec blocked(n) = 
      printfn "Blocking"
      // Only process the 'Resume' message
      mbox.Scan(fun msg ->
        match msg with
        // Return workflow to continue with
        | Resume -> Some(async {
            printfn "Resuming"
            return! processing(n) })
        // Other messages cannot be processed now
        | _ -> None)
        
    // Represents the active  state
    and processing(n) = async {
      printfn "Processing: %d" n
      // Process any message
      let! msg = mbox.Receive()
      match msg with
      | ModifyState(by) -> return! processing(n + by)
      | Resume -> return! processing(n)
      | Block -> return! blocked(n) }
    processing(0)
  )
  
// Sending messages from multiple threads 

open System
open System.Threading
  
// Thread performing calculations
let modifyThread() =
  let rnd = new Random(Thread.CurrentThread.ManagedThreadId)  
  while true do
    Thread.Sleep(500)
    // Send an update to the mailbox
    mbox.Post(ModifyState(rnd.Next(11) - 5)) 

let blockThread() =
  while true do
    Thread.Sleep(2000)
    mbox.Post(Block)    
    // Block the processing for one and half seconds
    Thread.Sleep(1500)
    mbox.Post(Resume) 

for proc in [ blockThread; modifyThread; modifyThread ] do
  Async.Start(async { proc() })


///////////////////////////////////////////////

let rec fib n = if n <= 2 then 1 else fib(n-1) + fib(n-2)
let fibs =  Async.Parallel[for i in 0..40 -> async{ return fib i } ]
            |> Async.RunSynchronously
fibs |> Array.iter (fun i -> printfn "%d")

let asyncTaskError = async { do! Async.Sleep 1000
                             failwith "Some Async Error"
                             return "Completed"  }

let asyncTask = async { do! Async.Sleep 1000 
                        return "Ok" }

let asyncTaskUnit = async { do! Async.Sleep 1000 
                            printfn "Async operation Completed" }

asyncTaskError  |> Async.Catch
                |> Async.RunSynchronously
                |> function
                    | Choice1Of2 result     ->  printfn "Async operation Completed %s" result
                    | Choice2Of2 (ex: exn)  ->  printfn "Async operation Error %s" ex.Message

let cancelHandler (ex:System.OperationCanceledException) = printfn "The Task has been cancelled"

asyncTaskUnit |> Async.Start
Async.TryCancelled(asyncTaskUnit, cancelHandler) |> Async.Start
Async.CancelDefaultToken()

let cancellationToken = new System.Threading.CancellationTokenSource()
Async.Start(asyncTaskUnit, cancellationToken.Token)

cancellationToken.Cancel()


Async.StartWithContinuations(asyncTaskError, 
                                (fun result ->   printfn "Async operation Completed %s" result),
                                (fun (ex:exn)   -> printfn "Async operation Error %s" ex.Message),
                                (fun (oce:System.OperationCanceledException)  -> printfn "The Task has been cancelled"))


////////////////////////////////////////////////////////

let readAsync (stream : System.IO.Stream) buffer offset count = 
    Async.FromBeginEnd( (fun (callback,state) -> stream.BeginRead(buffer,offset,count,callback,state)), 
                        (fun (asyncResult) -> stream.EndRead(asyncResult)))

////////////////////////////////////////////////////////

type msgTest =
    | Data of int
    | Fetch of int * AsyncReplyChannel<int>

let parallelWorker n f =
    MailboxProcessor.Start(fun inbox ->
                            let workers = Array.init n (fun _ -> MailboxProcessor.Start(f))
                            let rec loop i = async {
                                let! msg = inbox.Receive()
                                workers.[i].Post(msg)
                                return! loop((i+1)%n)
                            }
                            loop 0)

let agent = parallelWorker 4 (fun inbox ->
                                let rec loop() = async {
                                    let! (msg:msgTest) = inbox.Receive()
                                    match msg with
                                    | Data n -> do! Async.Sleep( 100 * n)                                                
                                    | Fetch (n, replay) -> let res = n * n
                                                           do! Async.Sleep( 100 * n )
                                                           replay.Reply(res)
                                    printfn "Thread id %d" System.Threading.Thread.CurrentThread.ManagedThreadId
                                    return! loop()
                                }
                                loop())


for i in [0..20] do match i % 2 with
                    | 0 -> agent.Post(Data i)
                    | _ -> //let taskAsync = agent.PostAndAsyncReply(fun f -> Fetch(i, f))// |> printfn "result %d"
                           Async.FromContinuations(agent.PostAndAsyncReply(fun f -> Fetch(i, f)), 
                                                    (fun (replay:int) -> printfn "result %d" replay),
                                                    (fun (error:exn)  -> printfn "error %s" error.Message),
                                                    (fun (cancelled:System.OperationCanceledException) -> printfn "cancelled"))
                                


////////////////////////////////////////////////////////////

let agentBatch f = MailboxProcessor<int>.Start(fun i ->
                let rec loop (c, lst) = async {
                    let! msg = i.Receive()
                    let newLst = msg::lst
                    if List.length newLst = 100 then
                        f(newLst)
                        return! loop (0, [])
                    return! loop ((c + 1), newLst) }
                loop (0, []))

let agent = agentBatch (fun newLst -> ignore(Async.RunSynchronously( Async.StartChild(async { newLst |> List.rev |> List.iter (fun i -> printfn "%d" i) }))))

for i in [1..1000] do agent.Post(i)

type BatchProcessingAgent<'T> (batchSize, timeout) =
    let batchEvent = new Event<'T[]>()
    let agent : MailboxProcessor<'T> = MailboxProcessor.Start(fun agent -> 
        let rec loop remainingTime messages = async {
            let start = DateTime.Now
            let! msg = agent.TryReceive(timeout = max 0 remainingTime)
            let elapsed = int (DateTime.Now - start).TotalMilliseconds
            match msg with 
            | Some(msg) when List.length messages = batchSize - 1 ->
                batchEvent.Trigger(msg :: messages |> List.rev |> Array.ofList)
                return! loop timeout []
            | Some(msg) ->
                return! loop (remainingTime - elapsed) (msg::messages)
            | None when List.length messages <> 0 -> 
                batchEvent.Trigger(messages |> List.rev |> Array.ofList)
                return! loop timeout []
            | None -> return! loop timeout [] }
        loop timeout [] )

    /// Triggered when the agent collects a group of messages
    member x.BatchProduced = batchEvent.Publish
    /// Send new message to the agent
    member x.Enqueue(v) = agent.Post(v)


open System 
open System.Drawing
open System.Windows.Forms

let frm = new Form()
let lbl = new Label(Dock = DockStyle.Fill)
frm.Controls.Add(lbl)
frm.Show()

// Create agent for bulking KeyPress events
let ag = new BatchProcessingAgent<_>(5, 5000)
frm.KeyPress.Add(fun e -> ag.Enqueue(e.KeyChar))
ag.BatchProduced
    |> Event.map (fun chars -> new String(chars))
    |> Event.scan (+) ""
    |> Event.add (fun str -> lbl.Text <- str)



// ----------------------------------------------------------------------------
// Blocking queue agent
// ----------------------------------------------------------------------------

open System
open System.Collections.Generic

type Agent<'T> = MailboxProcessor<'T>

type internal BlockingAgentMessage<'T> = 
  | Add of 'T * AsyncReplyChannel<unit> 
  | Get of AsyncReplyChannel<'T>

/// Agent that implements an asynchronous blocking queue
type BlockingQueueAgent<'T>(maxLength) =
  let agent = Agent.Start(fun agent ->
    
    let queue = new Queue<_>()

    let rec emptyQueue() = 
      agent.Scan(fun msg ->
        match msg with 
        | Add(value, reply) -> Some(enqueueAndContinue(value, reply))
        | _ -> None )
    and fullQueue() = 
      agent.Scan(fun msg ->
        match msg with 
        | Get(reply) -> Some(dequeueAndContinue(reply))
        | _ -> None )
    and runningQueue() = async {
      let! msg = agent.Receive()
      match msg with 
      | Add(value, reply) -> return! enqueueAndContinue(value, reply)
      | Get(reply) -> return! dequeueAndContinue(reply) }

    and enqueueAndContinue (value, reply) = async {
      reply.Reply() 
      queue.Enqueue(value)
      return! chooseState() }
    and dequeueAndContinue (reply) = async {
      reply.Reply(queue.Dequeue())
      return! chooseState() }
    and chooseState() = 
      if queue.Count = 0 then emptyQueue()
      elif queue.Count < maxLength then runningQueue()
      else fullQueue()

    // Start with an empty queue
    emptyQueue() )

  /// Asynchronously adds item to the queue. The operation ends when
  /// there is a place for the item. If the queue is full, the operation
  /// will block until some items are removed.
  member x.AsyncAdd(v:'T, ?timeout) = 
    agent.PostAndAsyncReply((fun ch -> Add(v, ch)), ?timeout=timeout)

  /// Asynchronously gets item from the queue. If there are no items
  /// in the queue, the operation will block unitl items are added.
  member x.AsyncGet(?timeout) = 
    agent.PostAndAsyncReply(Get, ?timeout=timeout)


// ----------------------------------------------------------------------------

let ag = new BlockingQueueAgent<int>(3)

async { 
  for i in 0 .. 10 do 
    do! ag.AsyncAdd(i)
    printfn "Added %d" i }
|> Async.Start

async { 
  while true do
    do! Async.Sleep(1000)
    let! v = ag.AsyncGet()
    printfn "Got %d" v }
|> Async.Start

///////////////////////////////////////////////////////////


// Stack OverFlow
let rec map(f : 'a -> 'b) (l : 'a list) =
    if l.IsEmpty then
        []
    else
        let t = List.tail l
        let h = List.head l
        let hb = f h
        let lb = map f t
        hb :: lb

    

let result = map (fun x -> x + x) [1..80000]
printfn "%A" result

// OK
let map(f : 'a -> 'b) (l : 'a list) =
    let rec map_cps(f : 'a -> 'b) (la : 'a list) (lb : 'b list) =
        if la.IsEmpty then
            List.rev lb
        else
            let t = List.tail la
            let h = List.head la
            let hb = f h
            let lb = hb :: lb
            map_cps f t lb
    map_cps f l []    

let result = map (fun x -> x + x) [1..80000]
printfn "%A" result

// Ok Refactor
let map(f : 'a -> 'b) (l : 'a list) =
    let rec map_cps(f : 'a -> 'b) (la : 'a list) (lb : 'b list) =
        match la with
        | h :: t -> map_cps f t (f h :: lb)
        | [] -> List.rev lb
    map_cps f l []

let result = map (fun x -> x + x) [1..80000]
printfn "%A" result

open System

type StringMonoid() = 
  member x.Combine(s1, s2) = String.Concat(s1, s2) 
  member x.Zero() = "" 
  member x.Yield(s) = s 
  member x.Delay(s) = s
 
let str = new StringMonoid() 
 
let hello = str { yield "Hello " 
                  yield "world!" }


//////////////////////////////////////////////////////////////


#r "System.Windows.Forms.DataVisualization.dll"

open System
open System.Drawing
open System.Windows.Forms
open System.Windows.Forms.DataVisualization.Charting

/// Add data series of the specified chart type to a chart
let addSeries typ (chart:Chart) =
    let series = new Series(ChartType = typ)
    chart.Series.Add(series)
    series

/// Create form with chart and add the first chart series
let createChart typ =
    let chart = new Chart(Dock = DockStyle.Fill, 
                          Palette = ChartColorPalette.Pastel)
    let mainForm = new Form(Visible = true, Width = 700, Height = 500)
    let area = new ChartArea()
    area.AxisX.MajorGrid.LineColor <- Color.LightGray
    area.AxisY.MajorGrid.LineColor <- Color.LightGray
    mainForm.Controls.Add(chart)
    chart.ChartAreas.Add(area)
    chart, addSeries typ chart

// ----------------------------------------------------------------------------
// Showing CPU usage

open System.Diagnostics

module CpuUsage =
  // Function that returns the current CPU usage
  let getCpuUsage = 
      let counter = 
          new PerformanceCounter
            ( CounterName = "% Processor Time",
              CategoryName = "Processor", InstanceName = "_Total" )
      (fun () -> counter.NextValue())

  // Create objects representing the chart. The tutorial uses a helper function 
  // createChart that is implemented in the previous tutorial. The following 
  // snippet configures the chart to use a spline area chart

  let chart, series = createChart SeriesChartType.SplineArea
  let area = chart.ChartAreas.[0]
  area.BackColor <- Color.Black
  area.AxisX.MajorGrid.LineColor <- Color.DarkGreen
  area.AxisY.MajorGrid.LineColor <- Color.DarkGreen
  chart.BackColor <- Color.Black
  series.Color <- Color.Green

  // The next snippet creates an asynchronous workflow that periodically 
  // updates the chart area. The implementation uses a while loop in the 
  // workflow and the workflow is started such that all user code runs 
  // on the main GUI thread

  let updateLoop = async { 
      while not chart.IsDisposed do
          let v = float (getCpuUsage()) 
          series.Points.Add(v) |> ignore
          do! Async.Sleep(250) }
  Async.StartImmediate updateLoop
  
  (*
  public partial class Form1 : Form 

{ 
    Thread t = null; 
    public Form1() 
    { 
        InitializeComponent(); 
        var actual = chart1.Series["Actual"]; 
        var area = chart1.ChartAreas["Default"]; 
        area.AxisY.Maximum = 100; 
        area.AxisY.Minimum = -100; 
    } 

    private void Form1_Load(object sender, EventArgs e) 
    { 
        t = new Thread(() => 
        { 
            ScrollChartDelegate del = new ScrollChartDelegate(ScrollChart); 

            for (int i = 0; i < 1000; i++) 
            { 
                //test series values 

                chart1.Invoke(del); 
                Thread.Sleep(200); 
            } 
        }); 
        t.Start(); 
    } 
    
    delegate void ScrollChartDelegate(); 
    
    private void ScrollChart() 
    { 
        var rand = new Random((int)DateTime.Now.Ticks); 
        chart1.Series["Actual"].Points.Add(new DataPoint(DateTime.Now.ToOADate(), rand.Next(-100, 100))); 
        chart1.ChartAreas["Default"].AxisX.Maximum = DateTime.Now.ToOADate(); 
        chart1.ChartAreas["Default"].AxisX.Minimum = DateTime.Now.AddSeconds(-10).ToOADate(); 
        chart1.Invalidate(); 
    } 

    private void Form1_FormClosing(object sender, FormClosingEventArgs e) 
    { 
        t.Abort(); 
    } 
} 
  *)

  

/////////////////////////////////////////


/// Huffman coding uses a binary tree whose leaves are the input symbols 
/// and whose internal nodes are the combined expected frequency of all the
/// symbols beneath them.
type HuffmanTree = 
    | Leaf of char * float
    | Node of float * HuffmanTree * HuffmanTree

/// Provides encoding and decoding for strings containing the given symbols and expected frequencies
type HuffmanCoder(symbols: seq<char>, frequencies : seq<float>) =
   
    /// Builds a list of leafs for a huffman encoding tree from the input frequencies
    let huffmanTreeLeafs =    
        Seq.zip symbols frequencies
        |> Seq.toList
        |> List.map Leaf
        
    /// Utility function to get the frequency from a huffman encoding tree node
    let frequency node =
        match node with
        | Leaf(_,p) -> p
        | Node(p,_,_) -> p    

    /// Builds a huffman encoding tree from a list of root nodes, iterating until a unique root node
    let rec buildCodeTree roots = 
        match roots |> List.sortBy frequency with
        | [] -> failwith "Cannot build a Huffman Tree for no inputs" 
        | [node] -> node
        | least::nextLeast::rest -> 
                   let combinedFrequency = frequency least + frequency nextLeast
                   let newNode = Node(combinedFrequency, least, nextLeast)
                   buildCodeTree (newNode::rest)
               
    let tree = buildCodeTree huffmanTreeLeafs
     
    /// Builds a table of huffman codes for all the leafs in a huffman encoding tree
    let huffmanCodeTable = 
        let rec huffmanCodes tree = 
            match tree with
            | Leaf (c,_) -> [(c, [])]
            | Node (_, left, right) -> 
                let leftCodes = huffmanCodes left |> List.map (fun (c, code) -> (c, true::code))
                let rightCodes = huffmanCodes right |> List.map (fun (c, code) -> (c, false::code))
                List.append leftCodes rightCodes
        huffmanCodes tree 
        |> List.map (fun (c,code) -> (c,List.toArray code))
        |> Map.ofList

    /// Encodes a string using the huffman encoding table
    let encode (str:string) = 
        let encodeChar c = 
            match huffmanCodeTable |> Map.tryFind c with
            | Some bits -> bits
            | None -> failwith "No frequency information provided for character '%A'" c
        str.ToCharArray()
        |> Array.map encodeChar
        |> Array.concat
       
    
    /// Decodes an array of bits into a string using the huffman encoding tree
    let decode bits =
        let rec decodeInner bitsLeft treeNode result = 
            match bitsLeft, treeNode with
            | [] , Node (_,_,_) -> failwith "Bits provided did not form a complete word"
            | [] , Leaf (c,_) ->  (c:: result) |> List.rev |> List.toArray
            | _  , Leaf (c,_) -> decodeInner bitsLeft tree (c::result)
            | b::rest , Node (_,l,r)  -> if b
                                         then decodeInner rest l result
                                         else decodeInner rest r result
        let bitsList = Array.toList bits
        new String (decodeInner bitsList tree [])
                 
    member coder.Encode source = encode source
    member coder.Decode source = decode source


////////////////////////

// Define a node type using a discriminated union
type Node =
    | InternalNode of int * Node * Node // an internal node with a weight and two subnodes
    | LeafNode of int * byte // a leaf with a weight and a symbol
 
/// Get the weight of a node (how many occurencies are in the input file)
let weight node =
    match node with
    | InternalNode(w,_,_) -> w
    | LeafNode(w,_) -> w
 
/// Creates the initial list of leaf nodes
let createNodes inputValues =
    let getCounts (leafNodes:(int*byte)[]) =
        inputValues |> Array.iter
            (fun b ->   let (w,v) = leafNodes.[(int)b]
                        leafNodes.[(int)b] <- (w+1,v))
        leafNodes
    [|for b in 0uy..255uy -> (0,b)|] |> getCounts
    |> List.ofArray
    |> List.map LeafNode
 
/// Create a Huffman tree using the initial list of leaf nodes as basis
let rec createHuffmanTree nodes =
    match nodes |> List.sort with
    | first :: second :: rest ->
        let newNode = InternalNode((weight first)+(weight second),first,second)
        createHuffmanTree (newNode::rest)
    | [node] -> node
    | [] -> failwith "Cannot create a huffman tree without input nodes"
 
/// Get a map of (symbol, length-of-huffman-code) pairs used when calculating the theoretical
/// length of the output.
let getHuffmanCodes topNode =
    let rec assignBitPattern node =
        match node with
        | LeafNode(_,v) -> [(v,[])]
        | InternalNode(_,leftNode,rightNode) ->
            let lCodes = assignBitPattern leftNode |> List.map (fun (v,c) -> (v,false::c))
            let rCodes = assignBitPattern rightNode |> List.map (fun (v,c) -> (v,true::c))
            List.append lCodes rCodes
    assignBitPattern topNode |> List.map (fun (v,(c:bool list)) -> (v,List.length c))
    |> Map.ofList
 
/// Calculates the theoretical size of the compressed data (without representing the
/// Huffman tree) in bits, and prints the result to stdout
let encode (input:byte[]) =
    let mapByte huffmanCodes b =
        match Map.tryFind b huffmanCodes with
        | Some huffmanCodeLength -> huffmanCodeLength
        | None -> failwith "Unknown input byte - invalid huffman tree"
    let huffmanCodes = createNodes input |> createHuffmanTree |> getHuffmanCodes
    let size = [|0|] // this is an ugly hack, but it is just for show and tell
    Array.iter (fun b -> size.[0] <- size.[0] + mapByte huffmanCodes b) input
    printfn "Original size      : %d bits" (input.Length*8)
    printfn "Compressed size    : %d bits" size.[0]
    
let fileName = @"c:\uncompressed_text.txt"
encode (File.ReadAllBytes(fileName))    


////////////////////////


 type AsyncSeqBuilder =
    // Waits for the result of a single asynchronous 
    // operation and then continues generating the sequence
    member Bind  : Async<'T> * ('T -> AsyncSeq<'U>) -> AsyncSeq<'U>
  
    // For every element of the input (asynchronous) sequence, 
    // yield all elements generated by the body of the for loop
    member For : AsyncSeq<'T> * ('T -> AsyncSeq<'TResult>) -> AsyncSeq<'TResult>
    member For : seq<'T>      * ('T -> AsyncSeq<'TResult>) -> AsyncSeq<'TResult>
 
   // Yield single/zero elements and concatenation of sequences
   member Yield : 'T   -> AsyncSeq<'T>
   member Zero  : unit -> AsyncSeq<'T>
   member Combine : AsyncSeq<'T> * AsyncSeq<'T> -> AsyncSeq<'T>
 
 type Microsoft.FSharp.Control.AsyncBuilder with
   // For every element of the input asynchronous sequence,
   // perform the specified asynchronous workflow
   member For : AsyncSeq<'T> * ('T -> Async<unit>) -> Async<unit>

///////////////////////////////////////////


type BinaryTree<'a> =     
    | Leaf of 'a     
    | Node of BinaryTree<'a> * BinaryTree<'a> 

let (|Node|Leaf|) (node : #System.Xml.XmlNode) =     
    if node.HasChildNodes then         
        Node (node.Name, { for x in node.ChildNodes -> x })     
    else         
        Leaf (node.InnerText


let printXml node =     
    let rec printXml indent node =         
        match node with         
        | Leaf (text) -> printfn "%s%s" indent text         
        | Node (name, nodes) -> printfn "%s%s:" indent name             
                                nodes |> Seq.iter (printXml (indent +"    "))     printXml "" node 


let (|ParseRegex|_|) re s =     
    let re = new System.Text.RegularExpressions.Regex(re)     
    let matches = re.Matches(s)     
    if matches.Count > 0 then         
        Some { for x in matches -> x.Value }  
    else       
    None let parse s =     


match s with     
        | ParseRegex "\d+" results -> printfn "Digits: %A" results     
        | ParseRegex "\w+" results -> printfn "Ids: %A" results     
        | ParseRegex "\s+" results -> printfn "Whitespace: %A" results     
        | _ -> failwith "known type" parse "hello world" parse "42 7 8" parse "\t\t\t" 

let rec printBinaryTreeValues t =     
match t with     
| Leaf x -> printfn "%i" x     
| Node (l, r) ->      
printBinaryTreeValues l 
printBinaryTreeValues r





type DirectoryRecord =  { Files: seq<string> 
                          Dir: string                          
                          Depth: int }

type DirectoryTree =     
    | DirectoryRecords of DirectoryRecord //* int
    | Dirs of seq<DirectoryTree>

let rec createDirectoryTree dir depth =
               seq { yield DirectoryRecords( {  Files = Directory.EnumerateFiles(dir, "*.*")
                                                Dir = dir
                                                Depth = depth } )
                     yield Dirs(seq { for d in Directory.EnumerateDirectories(dir) do yield! createDirectoryTree d (depth + 1) } )
                       }

let rec funcDirectoryTree (d:seq<DirectoryTree>, f:DirectoryRecord -> unit) =
    for o in d do 
        match o with     
        | DirectoryRecords dr -> f(dr)
        | Dirs s -> funcDirectoryTree(s, f)

let t = createDirectoryTree @"c:\Projects" 0

funcDirectoryTree(t, (fun rr -> printfn "Dir Name %s" rr.Dir))

    
//                     
//
//type DirectoryTree =     
//    | Records of DirectoryRecord * seq<DirectoryTree>
//
//let rec getAllFiles dir depth =
//               seq { yield Records( {   Files = Directory.EnumerateFiles(dir, "*.*")
//                                        Dir = dir
//                                        Depth = depth }, seq { for d in Directory.EnumerateDirectories(dir) do yield! getAllFiles d (depth + 1) } )
//                                        }
//                       
//
//
//
