 #light

 open System
 open System.Collections
 open System.Collections.Generic
 open System.Threading
 open Microsoft.FSharp.Control
 open System.Net
 open Microsoft.FSharp.Control.WebExtensions
 open System.IO
 open System.Net

 #load "ModuleUtils.fsx"

 Utilitues.ConversionUtil.intToString 4

(*There is a huge difference between STRING * STRING -> INT and  STRING -> STRING -> INT. Both are function types that take two string parameters and return an integer. However, one only accepts its argument in tuple form (STRING * STRING). This means that all parameters must be specified at the same time. The other function has had its arguments curried (STRING -> STRING -> INT), and so applying a single parameter results in a new function value.*)
 
 let (>>) f g x = g(f x)

 let x = Array.map >> Array.sum

 [<LiteralAttribute>]
 let ``THIS IS A CONSTANT`` = 10

 let lazy1 = Lazy<int>.Create(fun () -> printfn "Evaluatin x..."; 10)
 let lazy2 = lazy (printfn "Evaluating y..."; lazy1.Value + lazy1.Value)

 lazy2.Value
 
 let processes = 
    System.Diagnostics.Process.GetProcesses()
    |> Seq.head
                   
(*The ref type, sometimes referred to as a 􏰃􏰄􏰩 cell, allows you to store mutable data on the heap, enabling you to bypass limitations with mutable values that are stored on the stack. To retrieve the value of a ref cell, use the 􏰝 symbolic operator, and to set the value, use the := operator.   ref is not only the name of a type, but also the name of a function that produces ref values, which has the signature:
val ref: 'a -> 'a ref *)

(* In F#, you can use an object expression, which will create an anonymous class and return an instance of it for you. (The term “anonymous class” just means the compiler will generate the class for you and that you have no way to know its name.) This simplifies the process of creating one-time-use types, much in the same way using a lambda ex­ pression simplifies creating one-time-use functions. *)

let ObjectExpressionFunc = 
    let value = 5
    { new IDisposable with
        member x.Dispose() = 
            printfn "Value %i" value }
  
let makeResource name = 
   { new System.IDisposable 
     with member this.Dispose() = printfn "%s disposed" name }  
   
type System.Int32 with
    member this.ToHexString() = sprintf "8x%x" this
    
(*Modules in F# can be extended by simply creating a new module with the same name. As long as the new module’s namespace is opened, all new module functions, types, and values will be accessible.*)

open System.Collections.Generic
module List =
    let rec Skip n list =
        match n, list with
        | _, [] -> []
        | 0, list -> list
        | n, hd :: tl -> Skip (n - 1) tl



let arr1 = [| for i in 1..7 -> i |]
let arr2 = [| 1..7 |]
let len = 5
let arr3 = Array.init len (fun i -> i * i )
let arr4 = Array.zeroCreate<int> 4
let arr5 = Array2D.zeroCreate<float> 3 3
let arr6 = Array3D.zeroCreate<int> 4 4 3
let arr7 = 
        let jagged = Array.zeroCreate<int[]> 3
        jagged.[0] <- Array.init 2 (fun x -> x)
        jagged.[1] <- Array.init 3 (fun x -> x)
        jagged.[2] <- Array.init 1 (fun x -> x)
        jagged

try
    for i = 1 to 5 do printfn "%i" i
with
| :? StackOverflowException as ex -> ()
| :? OutOfMemoryException as ofm -> ()

let fold ls =
    ls
    |> List.map (fun i -> i.GetHashCode())
    |> List.fold (+) 0

type Point =
    val _x:int
    val _y:int

    new (x, y) = { _x = x; _y = y } 
    new (text:string) as this =
        let parts = text.Split([| ',' |])
        let (success, x) = Int32.TryParse(parts.[0])
        let (success, y) = Int32.TryParse(parts.[1])
        this._x = x
        this._y = y
         
type GenericClass<'a> = { Value:'a }

type Sandwich() =
    abstract Ingridients : string list
    default this.Ingridients = []

type BLT() =
    inherit Sandwich()
    override this.Ingridients = ["Bacon"; "Lattuce"]

type EmptySandwich() =
    inherit Sandwich()

// [<AbstractClassAttribute>]
// [<SealedAttribute>]
// [<StructAttribute>]

// :>  upcast  use to cast a class to a "base" interface
// :?> downcast

 // cap 3
 // cap 7 + ex 7-7, 7-8, 7-9, 7-17, 7-22, 7--24, 7-25, 9-9, 9-12, 10-4
 // f# 3 -> query{} , 
 

 (*When to Use a Struct Versus a Record
Because F# offers both structs and records as lightweight containers for data, it can be confusing to know when to use which.
The main benefit of using structs over records is the same benefit of using structs over classes—namely, that they offer different performance characteristics. When you’re dealing with a large number of small objects, allocating the space on the stack to create new structs is much faster than allocating many small objects on the heap. Similarly, with structs there is no additional garbage collection step, because the stack is cleared as soon as the function exits.
For the most part, however, the performance gain for struct allocation is negligible. In fact, if used thoughtlessly, structs can have a detrimental impact on performance due to excessive copying when passing structs as parameters. Also, the .NET garbage col­ lector and memory manager are designed for high-performance applications. When you think you have a performance problem, use the Visual Studio code profiler to iden­ tify what the bottleneck is first before prematurely converting all of your classes and records to structs.*)

let map f list =
    let rec mapTR f list acc =
        match list with
        |[] -> acc
        | hd :: tl -> mapTR f tl (f hd :: acc)
    mapTR f (List.rev list) []

(*Continuations
Imagine rather than passing the current state of the accumulator “so far” as a parameter to the next function call, you instead passed a function value representing the rest of the code to execute. That is, rather than storing “what’s left” on the stack, you store it in a function. This is known as the continuation passing style or simply using a continuation.
Continuations are function values that represent the rest of the code to execute when the current recursive call completes. This allows you to write tail-recursive functions even though there is more work to be done. Conceptually, you are trading stack space (the recursive calls) with heap space (the function values)*)

let printListRev list =
    let rec printListRevTR list cont =
        match list with
        | [] -> cont()
        | hd :: tl -> printListRevTR tl (fun () ->  printf "%d" hd 
                                                    cont())
    printListRevTR list (fun () -> printfn "Done!")

printListRev [1..10]

// Single-Case Active Pattern
let (|FileExtension|) filePath = System.IO.Path.GetExtension(filePath)
let res filePath =   
            match filePath with   
            | FileExtension ".jpg"
            | FileExtension ".png"
            | FileExtension ".gif" -> sprintf "image"
            | _ -> sprintf "None"

let (|ToUpper|) (x:string) = x.ToUpper()
printfn "%s" (ToUpper "ciao")

// Partial Active Pattern
let (|ToBool|_|) x =
    let (success, result) = Boolean.TryParse(x)
    if success then Some(result)
    else None

// Parameterized Active Pattern
let (|RegexMatch|_|) (pattern:string) (input:string) =
    let result = System.Text.RegularExpressions.Regex.Match(input, pattern)
    if result.Success then
        Some(result.Value)
    else
        None

// Multicase Pattern
(*Using a multicase active pattern, you can partition the input space into a known set of possible values. To define a multicase active pattern, simply use the banana clips again but include multiple identifiers to identify the categories of output*)
let (|Paragraph|Sentence|Space|) (input:string) =
    if input = " " then Space
    elif input = "/P" then Paragraph
    else Sentence



//let name = sprintf "Client %i" i
//let log msg = Console.WriteLine("{0}: {1}", name, msg)
    
let log msg = Console.WriteLine("{0}: {1}", DateTime.Now.ToLongTimeString(), msg)

let (<--) (m:'msg MailboxProcessor) x = m.Post x

let (<-!) (m:'msg MailboxProcessor) x = m.PostAndAsyncReply(x)



type AuctionMessage =
    | Offer of int * AuctionReply MailboxProcessor // Make a bid
    | Inquire o
    f AuctionReply MailboxProcessor     // Check the status    
and AuctionReply =
  | Status of int * DateTime // Asked sum and expiration
  | BestOffer                // Ours is the best offer
  | BeatenOffer of int       // Yours is beaten by another offer
  | AuctionConcluded of      // Auction concluded
                    AuctionReply MailboxProcessor * AuctionReply MailboxProcessor
  | AuctionFailed            // Failed without any bids
  | AuctionOver              // Bidding is closed

// A reusable parallel worker model built on F# agents
let parallelWorker n f = 
        MailboxProcessor.Start(fun inbox ->
            let workers = Array.init n (fun i -> MailboxProcessor.Start(f))
            let rec loop i = async {
                let! msg = inbox.Receive()
                workers.[i].Post(msg)
                return! loop ((i+1) % n) }
            loop 0 )

let agent f = 
        parallelWorker 8 (fun inbox ->
            let rec loop() = async {
                let! msg = inbox.Receive()
                f msg
                return! loop()
            }
            loop() )

////////////
//  WORKER AGENT
///////////
/// The internal type of messages used by BackgroundParallelWorker

type internal Message<'Index,'Data> = 
    | Request of 'Index * Async<'Data>         
    | Result of 'Index * 'Data                 
    | Clear                                    
                                               
/// A component that accepts a collection of jobs to run in the 
/// background and reports progress on these jobs.              
///                                                             
/// This component can be used from any thread with a synchronization
/// context, e.g. a GUI thread or an ASP.NET page handler. Events    
/// reporting progress are raised on the thread implied by the       
/// synchronization context, i.e. the GUI thread or the ASP.NET page 
/// handler.

type BackgroundParallelWorker<'Data>(jobs:seq<Async<'Data>>) =       

    let syncContext = SynchronizationContext.Current                 
    
    let raiseEventOnGuiThread (event:Event<_>) args =
        syncContext.Post(SendOrPostCallback(fun _ -> event.Trigger args),state=null)

    let changed   = new Event<_>()
    let completed = new Event<_>()

    // Check that we are being called from a GUI thread
    do match syncContext with 
        | null -> failwith "Failed to capture the synchronization context of the calling thread. The System.Threading.SynchronizationContext.Current of the calling thread is null"
        | _ -> ()

    let mbox = MailboxProcessor<_>.Start(fun mbox -> 
        let jobs = Seq.toArray jobs

        let rec ProcessMessages(results: Map<_,_>) = 
            // Read messages...
            async { let! msg = mbox.Receive()
                    match msg with 
                    | Result (i,v) -> 
                        // Update the 'results' set and process more messages
                        let results = AddResult(results,i,Some(v))           
                        return! ProcessMessages(results)                     
                    | Clear -> 
                        raiseEventOnGuiThread changed Map.empty
                        return! ProcessMessages(Map.empty)
                    | Request(i,job) -> 
                        // Spawn a request work item
                        do! Async.StartChild
                              (async { let! res = job
                                       do mbox.Post(Result(i,res)) }) |> Async.Ignore

                        // Update the 'results' set and process more messages
                        let results = AddResult(results,i,None)
                        return! ProcessMessages(results)  }
        and AddResult(results,i,v) =                       
            let results = results.Add(i,v)                 
            // Fire the 'results changed' event in the initial synchronization context
            raiseEventOnGuiThread changed results
            // Fire the 'completed' event in the initial synchronization context      
            if results.Count = jobs.Length && results |> Map.forall (fun _ v -> v.IsSome) then
                raiseEventOnGuiThread completed (results |> Map.map (fun k v -> v.Value))
            results
        ProcessMessages(Map.empty))

    member x.Start() =
        mbox.Post(Clear)
        for i,job in Seq.mapi (fun i x -> (i,x)) jobs do
            mbox.Post(Request(i,job))

    member x.ResultSetChanged = changed.Publish
    member x.ResultsComplete = completed.Publish


/////////////////////////////////////
type AsyncWorker<'T>(jobs) = 
    // Capture the synchronization context to allow us to raise events back on the GUI thread
    let syncContext = System.Threading.SynchronizationContext.Current

    // Check that we are being called from a GUI thread
    do match syncContext with                          
        | null -> failwith "Failed to capture the synchronization context of the calling thread. The System.Threading.SynchronizationContext.Current of the calling thread is null"
        | _ -> ()
        
    let allCompleted  = new Event<unit>()
    let error         = new Event<System.Exception>()
    let canceled      = new Event<System.OperationCanceledException>()
    let jobCompleted  = new Event<int * 'T>()

    let asyncGroup = new CancellationTokenSource() 

    let raiseEventOnGuiThread (event:Event<_>) args =
        syncContext.Post(SendOrPostCallback(fun _ -> event.Trigger args),state=null)

    member x.Start()    = 
        // Mark up the jobs with numbers             
        let jobs = jobs |> List.mapi (fun i job -> (job,i+1))
        let work =  
            Async.Parallel 
               [ for (job,jobNumber) in jobs do
                    yield async { let! result = job
                                  raiseEventOnGuiThread jobCompleted (jobNumber,result) } ]
             |> Async.Ignore
        Async.StartWithContinuations
            ( work,
              (fun res -> raiseEventOnGuiThread allCompleted res),
              (fun exn -> raiseEventOnGuiThread error exn),
              (fun exn -> raiseEventOnGuiThread canceled exn ),
              asyncGroup.Token)

    member x.CancelAsync(message) = 
        asyncGroup.Cancel()

    member x.JobCompleted  = jobCompleted.Publish
    member x.AllCompleted  = allCompleted.Publish
    member x.Canceled   = canceled.Publish       
    member x.Error      = error.Publish

//// TEST 
    /// Fetch the contents of a web page, asynchronously    
let httpAsync(url:string) =                             
        async { let req = WebRequest.Create(url)            
                let! resp = req.AsyncGetResponse()
                // rest is a callback
                use stream = resp.GetResponseStream() 
                use reader = new StreamReader(stream) 
                let text = reader.ReadToEnd() 
                return text }
let urls = 
        [ "http://www.live.com";                            
          "http://news.live.com";                           
          "http://www.yahoo.com";                           
          "http://news.yahoo.com";                          
          "http://www.google.com";                          
          "http://news.google.com"; ] 

let jobs =  [ for url in urls -> httpAsync url ]

let worker = new AsyncWorker<_>(jobs)

worker.JobCompleted.Add(fun (jobNumber, result) -> printfn "job %d completed with result %A" jobNumber result)
worker.AllCompleted.Add(fun () -> printfn "all done!" )
worker.Start()

////////// TEST SUM ARRAY
let sumArray (arr : int[]) =
    // Define a location in shared memory for counting
    let total = ref 0                                 
    let half = arr.Length/2
    Async.Parallel [ for (a,b) in [(0,half-1);(half,arr.Length-1)]  do 
                        yield async { let _ = for i = a to b do        
                                                  total := arr.[i] + !total
                                      return () } ]                        
      |> Async.Ignore                                                      
      |> Async.RunSynchronously                                            
    !total

sumArray [| 1;2;3 |]    
sumArray [| 1;2;3;4 |]  

let cancelToken = new System.Threading.CancellationTokenSource()
let cancelHandler (ex : System.OperationCanceledException) =
    printfn "Operation cancelled"

let sumArray2 (arr : int[]) =
    let half = arr.Length/2
    let parallelOps = Async.Parallel [ for (a,b) in [(0,half-1);(half,arr.Length-1)]  do 
                                          yield Async.TryCancelled(
                                                      (async { let total = ref 0                
                                                               let _ = for i = a to b do        
                                                                          total := arr.[i] + !total
                                                               return !total }), cancelHandler) ]
    Async.RunSynchronously(parallelOps, -1, cancelToken.Token)      
    |> Array.sum


sumArray2 [| 1;2;3 |]    
sumArray2 [| 1;2;3;4 |]    

////////////// CPS -> Continuation Passing Style ///////////

let rec factorial_k k = function
    | 0 | 1 -> k 1
    | n -> factorial_k (fun m -> k (n * m)) (n - 1)

let factorial = factorial_k (fun n -> n)

factorial 7

//////////  BINARY OF INT ///////////////
let binary_of_int n =
    [ for i in 8 * sizeof<int> - 1 .. -1 .. 0 ->
        if (n >>> i) % 2 = 0 then "0" else "1" ]
    |> String.concat ""

binary_of_int 9
1 <<< 8

////////////// ASYNC FIB ///////////////////////////
let rec asyncFib = function
    | 0 | 1 as n -> async { return n }
    | n -> async {  let! f = asyncFib(n - 2) |> Async.StartChild
                    let! n = asyncFib(n - 1)
                    let! m =f
                    return m + n }

let cancellationToken = new System.Threading.CancellationTokenSource()
Async.RunSynchronously(asyncFib 25, -1, cancellationToken.Token)

/////// CLEANING RESOURCES ///////////////////
let test wf =
    async { printfn "Start"
            use x = 
                { new System.IDisposable with
                    member x.Dispose() = 
                        printfn "Dispose" }
            do! wf
            printfn "End" }
    |> Async.Start

let asyncOpUnit = async { printfn "Sleeping"; do! Async.Sleep 1000 }

test asyncOpUnit
/////////////////////////////////////////

type Message =
    |Message1
    |Message2 of int
    |Message3 of string

let agentMessage = MailboxProcessor.Start(fun inbox ->
                        let rec loop n =
                            inbox.Scan(fun msg ->
                                            match msg with
                                            |Message1 ->    Some(async { printfn "Message1"
                                                                         return! loop(n+1) })
                                            |Message2 v ->    Some(async { printfn "Message2 - value %d" v
                                                                           return! loop(n+1) })
                                            |Message3 s ->    None ) 
                        loop 0 )

agentMessage.Post(Message1)
agentMessage.Post(Message2(9))
agentMessage.Post(Message3("ciao"))
agentMessage.Post(Message2(10))
agentMessage.CurrentQueueLength

/////////////// Module PARTITION LIST /////////////

  /// Partition elements of a list using the specified predicate.
  /// The result is a tuple containing elements (from the beginning 
  /// of the list that satisfy the predicate) and the rest of the list.
module List = 
    let partitionWhile f =
        let rec loop acc = function
            | x::xs when f x -> loop (x::acc) xs
            | xs -> List.rev acc, xs
        loop [] 

// Example use: Note that ' next' is not in the first
// part of the list, because it follows elements that do
// not start with a space!
[" foo"; " bar"; "goo"; "zoo"; " next"] |> List.partitionWhile (fun s -> s.StartsWith(" "))
                                   
/////////////// DOWNLOAD /////////////////

let download k (url:string) =
    let resp = System.Net.WebRequest.Create(url).GetResponse()
    use stream = resp.GetResponseStream()
    k stream

let reader_Stream k (stream:System.IO.Stream) =
    use reader = new System.IO.StreamReader(stream)
    k reader

let reader_String (reader:System.IO.TextReader) =
    reader.ReadToEnd()

let readGoogle = "http://www.google.com" |> download (reader_Stream reader_String)

let compressStream k (stream:System.IO.Stream) (mode:System.IO.Compression.CompressionMode) =
    use stream = new System.IO.Compression.GZipStream(stream, mode)
    k stream

let zipStream k (stream:System.IO.Stream) =
    compressStream k stream System.IO.Compression.CompressionMode.Compress

let unzipStream k (stream:System.IO.Stream) =
    compressStream k stream System.IO.Compression.CompressionMode.Decompress
// Protein Data Bank (PDB) - 1hxm rabbit serum haemopexin
"ftp://ftp.wwpdb.org/pub/pdb/data/structures/divided/XML/hx/1hxn.xml.gz" |> download (unzipStream (reader_Stream reader_String) )


////////////////////  SCRIPT REFERENCE /////////////////////

type Person(firstName, lastName) =
    member x.Name 
        with get() = sprintf "%s %s" firstName lastName

    new() = Person("Riccardo", "Terrell")

[<AbstractClassAttribute>]
type Animal() =
    abstract Speak : unit -> unit

[<InterfaceAttribute>]
type IWalk =
    abstract Walk : unit -> string
    abstract Fly : string

type ITest =
    abstract member Test : int * int -> string

type Test =
    member this.Test(x) = x * x
    interface ITest with
        member this.Test(x, y) = (x + y).ToString()

type ICarRentalService2 = 
    abstract member CalculatePrice: pickupDate:System.DateTime * returnDate:System.DateTime * pickupLocation:string * vehiclePreference:string -> float 
 
type CarRentalService2() = 
    interface ICarRentalService2 with 
        override this.CalculatePrice(pickupDate:System.DateTime, returnDate:System.DateTime, pickupLocation:string, vehiclePreference:string) = 5.5

type ICarRentalService = 
    abstract member CalculatePrice: System.DateTime * System.DateTime * string * string -> float 
 
type CarRentalService() = 
    interface ICarRentalService with 
        override this.CalculatePrice(pickupDate:System.DateTime, returnDate:System.DateTime, pickupLocation:string, vehiclePreference:string) = 5.5
 
type IPrintable =
    abstract member Print : string -> unit
    abstract member PrintInt : int -> int -> unit
   // abstract member PrintInt2 : (int * int) -> unit
 
type TestPrintable(v:string) =
    interface IPrintable with
        member x.Print(s) = printfn "%s - %s" v s
        member x.PrintInt v1 v2 = 
            let res = v1 + v2
            printfn "Result %d" res
                     
type TestPrintable2(v:string) =
    member x.Print(s) = (x :> IPrintable).Print(s)
    interface IPrintable with
        member x.Print(s) = printfn "%s - %s" v s
        member x.PrintInt v1 v2 = 
            let res = v1 + v2
            printfn "Result %d" res
                     
//let t = Test("Ciao")
//(t :> IPrintable).Print()
 
//Object expressions 
let print v =
    { new IPrintable with
        member x.Print(s) = printfn "%s - %s" v s
        member x.PrintInt v1 v2 = 
            let res = v1 + v2
            printfn "Result %d" res }

[<SealedAttribute>]
type Dog() =
    inherit Animal()
    override this.Speak() = printfn "Bau Bau"
    interface IWalk with
        member this.Walk() = "With 4 legs"
        member this.Fly = "No"

[<SealedAttribute>]
type Bird() =
    inherit Animal()
    override this.Speak() = printfn "Chip Chip"
    interface IWalk with
        member this.Walk() = "No"
        member this.Fly = "Yes"

let printHowWalks (x: 'a when 'a :> Animal) =
    let animal = x :> Animal
    animal.Speak()

[<StructAttribute>]
type PointStruct =
    val m_x : int
    val m_y : int
    new (x, y) = { m_x = x; m_y = y }

type myDel = delegate of int * int -> string

let myDelSample = myDel(fun x y -> string( x + y ))

let (|Even|Odd|) x =
    if x % 2 = 0 then Even
    else Odd

let isOddOrEven x =
    match x with
    | Even -> printfn "%d is Even" x
    | Odd -> printfn "%d is Odd" x

//////  GENERIC RECORD //////////////////
type genRec<'a, 'b> = {x:'a; y:'b}

let intStringRec = {x = 3.0; y = "three" }


////////////////////////
// type tree<'a> when 'a :> IComparable<'a> =
////////////////////////
 
let funTest2 (x:int, y:int) (f : int -> int -> int) =
    f x y
 
let funTest3 (x:int, y:int) (f : int * int -> int) =
    f(x, y)
 
let funTest (x:int) (f : int -> int -> int -> string) =
    let res = f x x x
    res.ToString()
 
let t = funTest 3 (fun x y z -> (x + y + z).ToString())
 
let t2 = funTest2(3, 4) (fun x y -> (x + y))
 
let myFun (x, y) = x + y
 
let testWithMyFun x y = funTest3(x, y) myFun
 
type Interface1 =
    abstract member Method1 : int -> int
 
type Interface2 =
    abstract member Method2 : int -> int
 
[<InterfaceAttribute>]
type Interface3 =
    inherit Interface1
    inherit Interface2
    abstract member Method3 : int -> int
 
type MyClass() =
    interface Interface3 with
        member this.Method1(n) = 2 * n
        member this.Method2(n) = n + 100
        member this.Method3(n) = n / 10
 
[<AbstractClassAttribute>]
type CircleShape(x:int) =
    member this.Radius with get() = x
    abstract Area : int with get
    abstract member AddToRadius: int -> int
    default this.AddToRadius(value) = this.Radius + value
 
type Cicle(x) =
    inherit CircleShape(x)
    override this.Area with get() = x * x * int(System.Math.PI)
 
 
type X() =
    member this.F([<System.ParamArrayAttribute>] args: System.Object[]) =
        for arg in args do
            printfn "%A" arg

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

let rec pairsList lst = 
    match lst with
    | x0::x1::xs -> (x0, x1)::pairsList(x1::xs)
    | [] | [_] -> [];;

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

let rec concatLists ls =
 match ls with  
 | head :: tail -> head @ (concatLists tail)
 | [] -> []

let listsToConcat = [[2;3;4];[6;8;9];[3;4;1;0]]

let list1' = [1;2;3;4;-4;-3;-2;-1]
let list2' = remove_if list1 (fun x -> (abs x &&&1) = 1)
let list3' = remove_if list1 (fun x -> (abs x &&&1) = 0)
let list4' = remove_if list1 (fun x -> x < 0)

let div = (/)
let add x y = x + y
let add' = fun x y -> x + y
let add'' x = fun y -> x + y
let add10'' = add'' 10

let mult x y = x * y
let mult5 x = 5 * x

let add10mult5 = add10'' >> mult5
add10mult5 6

// INLINE FUNC
let inline doSomething (x : ^a, y : ^b) =   
    let res = x + y
    res

let inline addInline x y = x = y

let rec isOdd n = (n = 1) || isEven (n - 1)
and isEven n = (n = 0) || isOdd (n - 1) 

let (===) str (regex:string) =
   System.Text.RegularExpressions.Regex.Match(str, regex).Success
let resReg = "El ricky is cool" === "El (.*) is cool"

let thisIsByte = 8uy
let thisIsShort = 16s
let thisIsUnsignedShort = 16us
let thisIsPointer = 9n
let thisIsUnsignedPointer = 9un

let ``this is a custom variable`` = 9

let (+*) a b = (a + b) * a * b // 1 +* 2

type Tree<'a> =
    | Node of Tree<'a> list
    | Value of 'a

[<MeasureAttribute>] type liter
[<MeasureAttribute>] type pint
let ratio = 1.0<liter> / 1.76<pint>
let convertPintToLiter pints = pints * ratio

//let macthV v =
//    match v with
//    | :? System.Int32 as x  -> ()
//    | :? System.Int16 as x -> ()
//    | _ -> ()

let allLineSeq path =
    use stream = System.IO.File.OpenText(path)
    seq { while not stream.EndOfStream do
            yield stream.ReadLine() }

type Drink =
    | Caffee of int 
    | Wine of int
    with
        member x.WhatAreYouDrinking() =
            match x with
            | Caffee v -> sprintf "%d Caffee" v
            | Wine v -> sprintf "%d Wine" v

let createLazy x = Lazy<int>.Create(fun () -> printfn "Evaukating x..."; x)

let ev = new Event<string>()
ev.Publish.Add(fun x -> printfn "%s" x)
ev.Trigger("ciao")

let filterEv = ev.Publish |> Event.filter(fun x -> x.StartsWith("c"))
Array.init
Array.zeroCreate 
Array2D.base1
let arr = new ResizeArray<string>()

type System.Net.Sockets.TcpListener with
    member x.AsycnAccept() =
        Async.FromBeginEnd(x.BeginAcceptTcpClient, x.EndAcceptTcpClient)


let ctx = System.ComponentModel.AsyncOperationManager.SynchronizationContext
let runCtx f = ctx.Post(new System.Threading.SendOrPostCallback(fun _ -> f()), null)

let context = System.Threading.ExecutionContext.Capture() 

let operation f (inbox:MailboxProcessor<_>) =
    let rec loop n = async {
            let! msg = inbox.Receive()
            f(msg)
            return! loop (n + 1) }
    loop 0

let opAgent f = MailboxProcessor.Start(operation f)

let currentDir = __SOURCE_DIRECTORY__
let currentFile = __SOURCE_FILE__

///////////////////////////////////////////////////////
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
///////////////////////////////////////////////////////
// Continuation passing functions

let printReverse lst =
    let rec printReverseLst lst' cont =
        match lst' with
        | [] -> cont()
        | hd :: tl -> printReverseLst tl (fun () -> printfn "%d " hd
                                                    cont() )
    printReverseLst lst (fun () -> printfn "Done!") 

///////////////////////////////////////////////////////
// Memoize
let memoize (f : 'a -> 'b) =
    let dict = new System.Collections.Generic.Dictionary<'a, 'b>()
    let memoizeFunc (input : 'a) =
        match dict.TryGetValue(input) with
        | true, x -> x
        | false, _ ->   let answer = f input
                        dict.Add(input, answer)
                        answer
    memoizeFunc

#nowarn "40"
let rec memoizeFib =
    let fib x =
        match x with
        | 0 | 1 -> 1
        | 2 -> 2
        | n -> memoizeFib(n - 1) + memoizeFib(n - 2)
    memoize fib
///////////////////////////////////////////////////////
// Constraints

let constraints' (x: 'a when 'a : (new : unit -> 'a)) =
    x

let constraints'' (x: 'a when 'a : (new : unit -> 'a) and 'a :> System.IComparable<'a>) =
    x

let constraints''' (x: 'a when 'a : struct) =
    x

let constraints'''' (x: 'a when 'a : not struct) =
    x

let constraints''''' (x: 'a when 'a :> System.Int16) =
    x
///////////////////////////////////////////////////////
// Events

type SetAction = Add | Remove

type SetActionEvent<'a>(value : 'a, action: SetAction) =
    inherit System.EventArgs()
    member x.Action = action
    member x.Value = value

type SetActionDelegate<'a> = delegate of obj * SetActionEvent<'a> -> unit

let evAction = new Event<SetActionDelegate<int>, SetActionEvent<int>>()
let evActionPublish = evAction.Publish

let resEv = evActionPublish |> Event.filter (fun setActionEve ->   match setActionEve.Action with
                                                                   | Add -> true
                                                                   | _ -> false)

resEv.Add(fun x -> printfn "Value %d" x.Value)

let partEv = evActionPublish |> Event.partition (fun setActionEve ->   match setActionEve.Action with
                                                                   | Add -> true
                                                                   | _ -> false)

(fst partEv).Add(fun _ -> printfn "ADD")
(snd partEv).Add(fun _ -> printfn "Remove")

evAction.Trigger(evAction, new SetActionEvent<int>(4, Add))
evAction.Trigger(evAction, new SetActionEvent<int>(4, Remove))

let form = new System.Windows.Forms.Form()
form.Width <- 600
form.Height <- 600
form.MouseMove 
    |> Event.filter(fun args -> args.X > 100)
    |> Event.add(fun args -> printfn "Coordinate %A %A" args.X args.Y)
form.Show()


///////////////////////////////////////////////////////

//let cancelToken = new System.Threading.CancellationTokenSource()
//let cancelHandler (ex : System.OperationCanceledException) =
//    printfn "Operation cancelled"

let asyncOperation = async { return (5 * 5) }

Async.StartWithContinuations(   asyncOperation, 
                                (fun result -> printfn "Result is %d" result),
                                (fun exn -> printfn "Error %s" exn.Message),
                                (fun can -> printfn "Operation cancelled"),
                                cancelToken.Token)

cancelToken.Cancel()

////// FUNCTION COMPOSITION >> /////////////////////////

let fun1 (value:int) : string =
    (value * value).ToString()

let fun2 (str:string) : int =
    let (res, value) = System.Int32.TryParse(str)
    if res then int( System.Math.Sqrt( float(value) ) )
    else failwith "impossible to cast"

let funComp = fun1 >> fun2

///////////////////////////////////////////////////////

let mapCol = Map.empty<int, string>
let mapCol1 = mapCol.Add(1, "Bugghina")
let mapCol2 = mapCol1.Add(2, "Ricky")

let lst1, lst2 = mapCol2 |> Map.toList |> List.unzip

lst1 |> List.iter (fun i -> printfn "%d" i)

/////////// OPTIONAL PARAMS ///////////////////////////

type optinalText(?text:string) =
    let text = defaultArg text "Bugghina"
    member x.Text 
        with get() = text

let opText = optinalText()
opText.Text

///////// EXTENSION METHOD ////////////////////////////

type System.Int32 with
    member i.IsPrime = 
        let lim = int (sqrt (float i))
        let rec check j =
            j > lim || (i % j <> 0 && check (j + 1))
        check 2

(4).IsPrime 

///////////////////////////////////////////////////////

let test'' = <@@ fun x -> x * x @@>
let test2'' = <@ 1 @>

let serInMem ex = 
    use mem = new System.IO.MemoryStream()
    (new System.Runtime.Serialization.Formatters.Binary.BinaryFormatter()).Serialize(mem, ex)
    mem.Flush()
    mem.ToArray()

let desInMem (b:byte[]) = 
    use mem = new System.IO.MemoryStream(b)
    let t = (new System.Runtime.Serialization.Formatters.Binary.BinaryFormatter()).Deserialize(mem)
    (t :?> Quotations.Expr)
             
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
let tc' = new TestGetComp(tc)
tc'.Get()

[<Sealed>]
type sealedtype() =
    member x.Data() =
     0

////////////////////////////////////////

//let rec revert_list l =   
//    match l with   
//    | [] -> []   
//    | x::rest -> (revert_list rest) @ [x]
//
//let rec insert_after elem newelem l =
//    match l with    
//    | [] -> [newelem]    
//    | x::rest -> if x = elem then                    
//                    (x::newelem::rest)                 
//                 else                     
//                    x::(insert_after elem newelem rest)
//
//let rec insert_before elem newelem l =    
//    match l with    
//    | [] -> [newelem]    
//    | x::rest -> if x = elem then                    
//                     (newelem::x::rest)                 
//                 else                    
//                     x::(insert_before elem newelem rest)        
//
//let list1 = [1;2;3;4;-4;-3;-2;-1]
//let list2 = insert_after 4 6 list1
//let list3 = insert_before 6 5 list2


//let rec remove_if l predicate =    
//    match l with    
//    | [] -> []    
//    | x::rest ->    if predicate(x) then                    
//                        (remove_if rest predicate)                 
//                    else                     
//                        x::(remove_if rest predicate)
//
//let list1 = [1;2;3;4;-4;-3;-2;-1]
//let list2 = remove_if list1 (fun x -> (abs x &&&1) = 1)
//let list3 = remove_if list1 (fun x -> (abs x &&&1) = 0)
//let list4 = remove_if list1 (fun x -> x < 0)
//
//printfn "%a" output_any list1
//printfn "%a" output_any list2
//printfn "%a" output_any list3
//printfn "%a" output_any list4

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
let rec isOdd2 n = (n = 1) || isEven2 (n - 1)
and isEven2 n = (n = 0) || isOdd2 (n - 1)

isOdd2(9)

////////////////////////////////////////
let (=~=) str (rgx:string) =
    System.Text.RegularExpressions.Regex.Match(str, rgx).Success

"Bugghina e' bella" =~= "Bugghina (.*)"

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
type PointStruct'(x:int, y:int) =
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

type SetAction2 = 
    | Added 
    | Removed

type SetOperationEventArgs<'a>(value : 'a, action : SetAction2) =
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
let cancelableTask = async {
                        printfn "Waiting 10 seconds..."
                        for i = 1 to 10 do 
                            printfn "%d..." i
                            do! Async.Sleep(1000)
                        printfn "Finished!" }
   
// Callback used when the operation is canceled
//let cancelHandler (ex : OperationCanceledException) = 
    //printfn "The task has been canceled."

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

//places |> List.map (fun (_ p) -> get(p));;
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

let memoizeWithDic (f: 'T -> 'U) =
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

//    match l with
//    | []    -> 0
//    | h::t -> h + sumList(t)

let rec aggregateList (op:int -> int -> int) init list =
    match list with
    | []    -> init
    | h::t  ->  let resultRest = aggregateList op init list
                op resultRest h

let add'2 a b = a + b
let mul'2 a b = a * b

aggregateList add 0 [1..5]

let rec last l =
    match l with
    | [] -> []
    | [h] -> h
    | h::t -> last t

//let rec mapAcc f inputList acc =
//    match inputList with
//    | [] -> List.rev acc
//    | h::t mapAcc f t (f h :: acc)


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
                                                            if n=0 then cont results  })))
    
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

/////////////////////// DECISION TREE /////////////////////

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

// Decision tree for testing clients

// Root node on level 1
let rec tree = 
    Query({ Title = "More than $40k" 
            Test = (fun cl -> cl.Income > 40000)
            Positive = moreThan40; 
            Negative = lessThan40 })
// First option on the level 2
and moreThan40 = 
    Query({ Title = "Has criminal record"
            Test = (fun cl -> cl.CriminalRecord)
            Positive = Result("NO"); 
            Negative = Result("YES") })
// Second option on the level 2
and lessThan40 = 
    Query({ Title = "Years in job"
            Test = (fun cl -> cl.YearsInJob > 1)
            Positive = Result("YES"); 
            Negative = usesCredit })
// Additional question on level 3
and usesCredit = 
    Query({ Title = "Uses credit card"
            Test = (fun cl -> cl.UsesCreditCard)
            Positive = Result("YES"); 
            Negative = Result("NO") })

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
//  let toRectangleF(rc) = 
//    RectangleF(rc.Left, rc.Top, rc.Width, rc.Height);;
    
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
let addCache = 
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
addCache(2, 3)

// Value is obtained from the cache
addCache(2, 3)

// Section Reusable memoization

// Generic memoization function
let memoizeGen(f) =    
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
let map'' f ls =
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

let tree' = 
  Node(Node(Node(Leaf(5), Leaf(8)), Leaf(2)),
       Node(Leaf(2), Leaf(9)));;
sumTree(tree')

//// Deep recursion causes stack overflow!
//let imbalancedTree =
//  test |> List.fold_left(fun st v ->
//      // Add node with previous tree on the right
//      Node(Leaf(v), st)) (Leaf(0))
//sumTree(imbalancedTree)

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
//sumTreeCont imbalancedTree (fun r ->
//  printfn "Result is: %d" r)
//
//// Returning sum from the continuation
//sumTreeCont imbalancedTree (fun a -> a)
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
        
let allFilesSeq dir =
    seq { for file in System.IO.Directory.GetFiles(dir) do
            let creationTime = System.IO.File.GetCreationTime(file)
            yield (file,creationTime)
        }


// Message passing concurrency

type MessageMbox = 
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

//let rec fib n = if n <= 2 then 1 else fib(n-1) + fib(n-2)
let fibs =  Async.Parallel[for i in 0..40 -> async{ return fib i } ]
            |> Async.RunSynchronously
fibs |> Array.iter (fun i -> printfn "%d" i)

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

//let cancelHandler (ex:System.OperationCanceledException) = printfn "The Task has been cancelled"

asyncTaskUnit |> Async.Start
Async.TryCancelled(asyncTaskUnit, cancelHandler) |> Async.Start
Async.CancelDefaultToken()

//let cancellationToken = new System.Threading.CancellationTokenSource()
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

let parallelWorkerPar n f =
    MailboxProcessor.Start(fun inbox ->
                            let workers = Array.init n (fun _ -> MailboxProcessor.Start(f))
                            let rec loop i = async {
                                let! msg = inbox.Receive()
                                workers.[i].Post(msg)
                                return! loop((i+1)%n)
                            }
                            loop 0)

let agentPar = parallelWorker 4 (fun inbox ->
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


//for i in [0..20] do 
//    match i % 2 with
//    | 0 -> agentPar.Post(Data i)
//    | _ -> //let taskAsync = agent.PostAndAsyncReply(fun f -> Fetch(i, f))// |> printfn "result %d"
//            Async.FromContinuations(agentPar.PostAndAsyncReply(fun f -> Fetch(i, f)), 
//                                    (fun (replay:int) -> printfn "result %d" replay),
//                                    (fun (error:exn)  -> printfn "error %s" error.Message),
//                                    (fun (cancelled:System.OperationCanceledException) -> printfn "cancelled"))
                                


////////////////////////////////////////////////////////////

let agentBatch f = MailboxProcessor<int>.Start(fun i -> 
                                                let rec loop (c, lst) = async {
                                                    let! msg = i.Receive()
                                                    let newLst = msg::lst
                                                    if List.length newLst = 100 then
                                                        f(newLst)
                                                        return! loop (0, [])
                                                    return! loop ((c + 1), newLst) }
                                                loop (0, []) )

let agentB2 = agentBatch (fun newLst -> ignore(Async.RunSynchronously( Async.StartChild(async { newLst |> List.rev |> List.iter (fun i -> printfn "%d" i) }))))

for i in [1..1000] do agentB2.Post(i)

type BatchProcessingAgent<'T> (batchSize, timeout) = 
    let batchEvent = new Event<'T[]>()
    let agent : MailboxProcessor<'T> = MailboxProcessor.Start(fun agent ->
        let rec loop remainingTime messages = async {
            let start = DateTime.Now
            let! msg = agent.TryReceive(timeout = max 0 remainingTime)
            let elapsed = int (DateTime.Now - start).TotalMilliseconds
            match msg with
            | Some(msg) when List.length messages = batchSize - 1 ->
                batchEvent.Trigger(msg :: messages |> List.rev |> Array.ofList)
                return! loop timeout []
            | Some(msg) ->
                return! loop (remainingTime - elapsed) (msg::messages)
            | None when List.length messages <> 0 ->
                batchEvent.Trigger(messages |> List.rev |> Array.ofList)
                return! loop timeout []
            | None -> return! loop timeout [] }
        loop timeout [] )

    /// Triggered when the agent collects a group of messages
    member x.BatchProduced = batchEvent.Publish
    /// Send new message to the agent
    member x.Enqueue(v) = agent.Post(v)


open System.Drawing
open System.Windows.Forms

let frm = new Form()
let lbl = new Label(Dock = DockStyle.Fill)
frm.Controls.Add(lbl)
frm.Show()

// Create agent for bulking KeyPress events
let ag = new BatchProcessingAgent<_>(5, 5000)
frm.KeyPress.Add(fun e -> ag.Enqueue(e.KeyChar))
ag.BatchProduced
    |> Event.map (fun chars -> new String(chars))
    |> Event.scan (+) ""
    |> Event.add (fun str -> lbl.Text <- str)



// ----------------------------------------------------------------------------
// Blocking queue agent
// ----------------------------------------------------------------------------

type Agent<'T> = MailboxProcessor<'T>

type internal BlockingAgentMessage<'T> =
    | Add of 'T * AsyncReplyChannel<unit>
    | Get of AsyncReplyChannel<'T>

/// Agent that implements an asynchronous blocking queue
type BlockingQueueAgent<'T>(maxLength) =
    let agent = Agent.Start(fun agent ->
        let queue = new Queue<_>()

        let rec emptyQueue() =
          agent.Scan(fun msg ->
            match msg with
            | Add(value, reply) -> Some(enqueueAndContinue(value, reply))
            | _ -> None )
        and fullQueue() =
            agent.Scan(fun msg ->
            match msg with
            | Get(reply) -> Some(dequeueAndContinue(reply))
            | _ -> None )
        and runningQueue() = async {
            let! msg = agent.Receive()
            match msg with
            | Add(value, reply) -> return! enqueueAndContinue(value, reply)
            | Get(reply) -> return! dequeueAndContinue(reply) }

        and enqueueAndContinue (value, reply) = async {
            reply.Reply()
            queue.Enqueue(value)
            return! chooseState() }
        and dequeueAndContinue (reply) = async {
            reply.Reply(queue.Dequeue())
            return! chooseState() }
        and chooseState() =
            if queue.Count = 0 then emptyQueue()
            elif queue.Count < maxLength then runningQueue()    
            else fullQueue()

    // Start with an empty queue
        emptyQueue() )

    /// Asynchronously adds item to the queue. The operation ends when
    /// there is a place for the item. If the queue is full, the operation
    /// will block until some items are removed.
    member x.AsyncAdd(v:'T, ?timeout) =
        agent.PostAndAsyncReply((fun ch -> Add(v, ch)), ?timeout=timeout)

    /// Asynchronously gets item from the queue. If there are no items
    /// in the queue, the operation will block unitl items are added.
    member x.AsyncGet(?timeout) =
        agent.PostAndAsyncReply(Get, ?timeout=timeout)


// ----------------------------------------------------------------------------

let agb = new BlockingQueueAgent<int>(3)

async {
    for i in 0 .. 10 do
        do! agb.AsyncAdd(i)
        printfn "Added %d" i }
|> Async.Start

async {
    while true do
        do! Async.Sleep(1000)
        let! v = agb.AsyncGet()
        printfn "Got %d" v }
|> Async.Start


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

///////////////////////////////////////////
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

let t'' = createDirectoryTree @"c:\Projects" 0

funcDirectoryTree(t'', (fun rr -> printfn "Dir Name %s" rr.Dir))

//type DirectoryTree =     
//    | Records of DirectoryRecord * seq<DirectoryTree>
//
//let rec getAllFiles dir depth =
//               seq { yield Records( {   Files = Directory.EnumerateFiles(dir, "*.*")
//                                        Dir = dir
//                                        Depth = depth }, seq { for d in Directory.EnumerateDirectories(dir) do yield! getAllFiles d (depth + 1) } )
//                                        }

//////////////////////////////////////////////////////////

let synchronize f = 
  let ctx = System.Threading.SynchronizationContext.Current 
  f (fun g arg ->                                           
    let nctx = System.Threading.SynchronizationContext.Current 
    if ctx <> null && ctx <> nctx then ctx.Post((fun _ -> g(arg)), null)
    else g(arg) )

let evAsync = new Event<int>()
let evAsyncPub = evAsync.Publish

let rec asyncLoopObservable(count) f = async {
    let! args = Async.AwaitEvent(evAsyncPub)
    f(args)
    return! asyncLoopObservable(count + 1) f }

Async.Start( asyncLoopObservable(0) (fun f -> printfn "value %d" f) )

evAsync.Trigger(9)    

//////////////////////////////////////////////////////////

let rec doPermute(inChars:char array, out:System.Text.StringBuilder, used:bool array, len:int, level:int) =
    if level = len then 
        printfn "%s" (out.ToString())
    else
        for i in [0 .. len - 1] do
            if not( used.[i] ) then
                out.Append( inChars.[i] ) |> ignore
                used.[i] <- true
                doPermute(inChars, out, used, len, level + 1)
                used.[i] <- false
                out.Length <- (out.Length - 1)

let permute (str:string) =
    let len = str.Length
    let used = Array.init len (fun _ -> false)
    let out = new System.Text.StringBuilder()
    let inChars = str.ToCharArray()
    doPermute(inChars, out, used, len, 0)

///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////


#load "Threading\Pseq.fs"
open System
open System.IO
open Microsoft.FSharp.Collections

let rec getFiles dirPath:seq<string> = seq {
    for file in Directory.EnumerateFiles(dirPath, "*.*") do yield file
    for dir in Directory.EnumerateDirectories(dirPath, "*.*") do yield! getFiles dir }

let calcSize (size:int64) = //"{0:##.##}"  {0:0.00}
    let str v n = String.Format("{0:##.##} {1}", v.ToString(), n)
    let calc x y = (float(x) / float(y))      
    if size > 11252401998659584L then 
        let res = calc size 11252401998659584L
        //let s = (str (res.ToString()) "PetaByte")
        (res, (str res "PetaByte"))
    elif size > 10988673826816L then 
        let res = (calc size 10988673826816L)
        (res, (str  res "TeraByte")) 
    elif size > 1073741824L then 
        let res = (calc size 1073741824L)
        (res, (str res "GygaByte"))
    elif size > 1048576L then 
        let res = (calc size 1048576L)
        (res, (str res "MegaByte"))
    else 
        let res = (calc size 1024L)
        (res, (str res "KiloByte"))

let getSizeDir dirPath =
    (getFiles dirPath)
    |> PSeq.map (fun file -> new System.IO.FileInfo(file))
    |> PSeq.map (fun file -> file.Length)
    |> PSeq.sum
    |> calcSize

let dirSize dirPath =
        sprintf "The Directory %s size is %s" dirPath (getSizeDir dirPath)

let megabyte = 1024L * 1024L
let gigabyte = megabyte * 10234L
let terabyte = gigabyte * 1024L
let petabyte = terabyte * 1024L
let exabyte = petabyte * 1024L
let zatabyte = exabyte * 1024L


let dirDev = @"T:\"
// let sumSeq sequence1 = Seq.fold (fun acc elem -> acc + elem) 0 sequence1

let perAction dirDev =
    let sw = System.Diagnostics.Stopwatch.StartNew()
    let s =  Directory.EnumerateDirectories(dirDev, "*", SearchOption.TopDirectoryOnly) |> Seq.truncate 6 |> Seq.fold (fun acc f -> let (size, str) = getSizeDir f
                                                                                                                                    printfn "The Directory %s size is %s" f str
                                                                                                                                    acc + size) 0.0
    printfn "Operation completed in %d" sw.ElapsedMilliseconds
    printfn "Total Size %s" (s.ToString())
perAction @"T:\"
let s = 1111111L
let b = 1048576L
//The Directory T:\Work size is 42.000000 MegaByte
//The Directory T:\TekCode size is 1.000000 GigaByte
//The Directory T:\Temp size is 1.000000 MegaByte
//The Directory T:\PsTools size is 7.000000 MegaByte
//The Directory T:\SourceCode_Book size is 8.000000 GigaByte
//The Directory T:\MyCodeTools size is 696.000000 MegaByte
//The Directory T:\MyMac size is 507.000000 KiloByte
//The Directory T:\PersonalInfo size is 69.000000 MegaByte
//The Directory T:\Projects size is 6.000000 GigaByte
//The Directory T:\MSDN Certification size is 9.000000 GigaByte
//The Directory T:\Docs size is 18.000000 GigaByte
//The Directory T:\Developer Tools size is 1.000000 GigaByte
//The Directory T:\CodeToCheck size is 39.000000 MegaByte
//The Directory T:\Analyzer Tools size is 37.000000 MegaByte
//The Directory T:\CODE size is 40.000000 GigaByte
//The Directory T:\DataTest size is 1.000000 GigaByte


let synchronize f = 
  let ctx = System.Threading.SynchronizationContext.Current 
  f (fun g arg ->
    let nctx = System.Threading.SynchronizationContext.Current 
    if ctx <> null && ctx <> nctx then ctx.Post((fun _ -> g(arg)), null)
    else g(arg) )

type Microsoft.FSharp.Control.Async with 
  static member AwaitObservable(ev1:IObservable<'a>) =
    synchronize (fun f ->
      Async.FromContinuations((fun (cont,econt,ccont) -> 
        let rec callback = (fun value ->
          remover.Dispose()
          f cont value )
        and remover : IDisposable  = ev1.Subscribe(callback) 
        () )))

let semaphoreStates2() = async {
    while true do
      let! md = Async.AwaitObservable(this.MouseLeftButtonDown)
      display(green) 
      let! md = Async.AwaitObservable(this.MouseLeftButtonDown)
      display(orange) 
      let! md = Async.AwaitObservable(this.MouseLeftButtonDown)
      display(red)  
    }
  do
    semaphoreStates2() |> Async.StartImmediate
