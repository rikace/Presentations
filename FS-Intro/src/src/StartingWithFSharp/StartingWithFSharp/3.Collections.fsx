#r @"..\packages\FSharp.Collections.ParallelSeq.1.0.2\lib\net40\FSharp.Collections.ParallelSeq.dll"

open System
open FSharp.Collections.ParallelSeq

module Arrays = 
    
    /// The empty array
    let array1 = [||]
    
    let array2 = [| "hello"; "world"; "and"; "hello"; "world"; "again" |]
    let array3 = [| 1..1000 |]

    [| for i in 0 .. 3 -> (i,i*i) |]

    let first_ten = [|1..10|] 
    let evens = [|2..2..10|]
    let chars = [|'a'..'z'|]

    ["apples"; "oranges"; "pumpkins"; "pomegranates"; "kiwi"] @ ["bears"; "tigers"; "kiwi"; "lions"; "penguins"]


    let number4 = array3.[3]

    /// An array containing only the words "hello" and "world"
    let array4 = 
        [| for word in array2 do
               if word.Contains("l") then yield word |]
    
    /// An array initialized by index and containing the even numbers from 0 to 2000
    let evenNumbers = Array.init 1001 (fun n -> n * 2)
    
    /// sub-array extracted using slicing notation
    let evenNumbersSlice = evenNumbers.[0..500]
    
    for word in array4 do
        printfn "word: %s" word

    // modify an array element using the left arrow assignment operator
    array2.[1] <- "WORLD!"
    
    /// Calculates the sum of the lengths of the words that start with 'h'
    let sumOfLengthsOfWords = 
        array2
        |> Array.filter (fun x -> x.StartsWith "h")
        |> Array.sumBy (fun x -> x.Length)


  // With Array.zeroCreate:
    let emptyStrings : string[] = Array.zeroCreate 100

    let emptyStrings' = Array.zeroCreate<System.IO.FileInfo> 100

    // With Array.init:
    let lastDays' year =
       Array.init 12 (fun i ->
          let month = i+1
          let firstDay = DateTime(year, month, 1)
          let lastDay = firstDay.AddDays(float(DateTime.DaysInMonth(year, month)-1))
          lastDay )

    printfn "%A" (lastDays' 2014) 

    //#time

    [|1..100|]
    |> Array.map(fun _ ->   let arr = Array2D.init 1000 1000 (fun x y -> 0)
                            for i = 0 to 999 do
                                for j = 0 to 999 do
                                    arr.[i,j] <-  arr.[i,j] + 1
                            arr) |> ignore   

    [|1..100|]
    |> Array.Parallel.map(fun _ ->  let arr = Array2D.init 1000 1000 (fun x y -> 0)
                                    for i = 0 to 999 do
                                        for j = 0 to 999 do
                                            arr.[i,j] <-  arr.[i,j] + 1
                                    arr) |> ignore   

module Lists = 
    let list1 = [] /// an empty list
    
    let list2 = [ 1; 2; 3 ] /// list of 3 elements
    
    let list3 = 42 :: list2 /// a new list with '42' added to the beginning
    
    let numberList = [ 1..1000 ] /// list of integers from 1 to 1000
    
    // A list of the numbers from 0 to 99
    let sampleNumbers = [ 0..99 ]
   
    let intArray = [|1;2;3|]
    
    let intList =[1..100]
        
    let list = [1..100]
    
    [ for i in 0 .. 3 -> (i,i*i) ]

    let squares = [for i in 1..100 do yield i*i]  
    
    let even = [for i in 1..100 do if i%2=0 then yield i]

    let square x = x * x
    // You can use parens to clarify precedence. In this example,
    // do "map" first, with two args, then do "sum" on the result.
    // Without the parens, "List.map" would be passed as an arg to List.sum
    let sumOfSquaresTo100 =
       List.sum ( List.map square [1..100] )

    // You can pipe the output of one operation to the next using "|>"
    // Here is the same sumOfSquares function written using pipes
    let sumOfSquaresTo100piped =
       [1..100] |> List.map square |> List.sum  // "square" was defined earlier

    // you can define lambdas (anonymous functions) using the "fun" keyword
    let sumOfSquaresTo100withFun =
       [1..100] |> List.map (fun x->x*x) |> List.sum

    // In F# there is no "return" keyword. A function always
    // returns the value of the last expression used.

    /// A list containing all the days of the year
    let daysList = 
        [ for month in 1..12 do
              for day in 1..System.DateTime.DaysInMonth(2012, month) do
                  yield System.DateTime(2012, month, day) ]
    
    /// A list containing the tuples which are the coordinates of the black squares on a chess board.
    let blackSquares = 
        [ for i in 0..7 do
              for j in 0..7 do
                  if (i + j) % 2 = 1 then yield (i, j) ]
    
    /// Square the numbers in numberList, using the pipeline operator to pass an argument to List.map   
    let squares' = numberList |> List.map (fun x -> x * x)

    
    /// Computes the sum of the squares of the numbers divisible by 3.
    let sumOfSquaresUpTo n = 
        numberList
        |> List.filter (fun x -> x % 3 = 0)
        |> List.sumBy (fun x -> x * x)

module Sequences = 
    // Sequences are evaluated on-demand and are re-evaluated each time they are iterated.
    // An F# sequence is an instance of a System.Collections.Generic.IEnumerable<'T>,
    // so Seq functions can be applied to Lists and Arrays as well.
    /// The empty sequence
    let seq1 = Seq.empty
    
    seq { for i in 0 .. 3 -> (i,i*i) }

    let seq2 = seq {    yield "hello"
                        yield "world"
                        yield "and"
                        yield "hello"
                        yield "world"
                        yield "again" }
    
    let numbersSeq = seq { 1..1000 }
    
    /// another array containing only the words "hello" and "world"
    let seq3 = 
        seq { for word in seq2 do
                if word.Contains("l") then yield word }
    
    let evenNumbers = Seq.init 1001 (fun n -> n * 2)
    let rnd = System.Random()
    
    /// An infinite sequence which is a random walk
    //  Use yield! to return each element of a subsequence, similar to IEnumerable.SelectMany.
    let rec randomWalk x = 
        seq { 
            yield x
            yield! randomWalk (x + rnd.NextDouble() - 0.5)
        }
    
    let first100ValuesOfRandomWalk = 
        randomWalk 5.0
        |> Seq.truncate 100
        |> Seq.toList

    // Simple lit comprehenion
    let numbersNear x =
        [
            yield x - 1
            yield x
            yield x + 1
        ]
    
    // More complex list comprehensions
    let x =
        [   let negate x = -x
            for i in 1 .. 10 do
                if i % 2 = 0 then
                    yield negate i
                else
                    yield i ]
                
    // Generate the first ten multiples of a number
    let multiplesOf x = [ for i in 1 .. 10 do yield x * i ]

    // Simplified list comprehension
    let multiplesOf2 x = [ for i in 1 .. 10 -> x * i ]
                
    // List comprehension for prime numbers
    let primesUnder max =
        [
            for n in 1 .. max do
                let factorsOfN =
                    [ 
                        for i in 1 .. n do
                            if n % i = 0 then
                                yield i 
                    ]
            
                // n is prime if its only factors are 1 and n
                if List.length factorsOfN = 2 then
                    yield n
        ]

    // From a range:
    let integersRange = {1..1000}

    // From a sequence expression:
    let integersExpression = 
       seq { 
          for i in 1..1000 do
             yield i
       }

    // From a sequence expression (short form):
    let integersExpression2 = 
       seq { 
          for i in 1..1000 -> i
       }

    // Using Seq.init:
    let integers = Seq.init 1000 (fun i -> i + 1)

    // Using Seq.initInfinite:
    let integers' = Seq.initInfinite (fun i -> i + 1)    
                
module Dictionaries =

    open System.Collections.Generic

    // Basic dictionary operations

    type LatLong = { Lat : double; Long : float }

    // Creating a dictionary:
    let zipLocations = Dictionary<int, LatLong>()

    // Adding items using add:
    zipLocations.Add(11373, {Lat = 40.72; Long = -73.87})
    zipLocations.Add(11374, {Lat = 40.72; Long = -73.86})

    // Adding items using indexed assignment:
    zipLocations.[11375] <- {Lat = 40.72; Long = -73.84}
    zipLocations.[11377] <- {Lat = 40.74; Long = -73.9}

    // Retrieving items:
    printfn "%A" zipLocations.[11377]

    // Updating items using assignment:
    zipLocations.[11377] <- {Lat = 40.75; Long = -74.2}
    printfn "%A" zipLocations.[11377]

    // Adding a duplicate key/value (causes an exception):
    zipLocations.Add(11374, {Lat = 40.72; Long = -73.86})

    // Iterating over the dictionary by treating it as a sequence:
    zipLocations
    |> Seq.iter (fun kvp -> 
       printfn "%i %f %f" kvp.Key kvp.Value.Lat kvp.Value.Long)

    // Iterating over the keys as a sequence:
    zipLocations.Keys
    |> Seq.iter (fun key -> printfn "%i" key)

    // Iterating over the values as a sequence:
    zipLocations.Values
    |> Seq.iter (fun value -> printfn "%f %f" value.Lat value.Long)
        

    let zipLocations' =
       [
          11373, {Lat = 40.72; Long = -73.87}
          11374, {Lat = 40.72; Long = -73.86}
          11375, {Lat = 40.72; Long = -73.84}
          11377, {Lat = 40.74; Long = -73.9}
       ] |> dict

    printfn "%A" zipLocations'.[11374]       


    let map1 = Map.ofList [ (1, "One"); (2, "Two"); (3, "Three") ]
    let map2 = map1 |> Map.map (fun key value -> value.ToUpper())
    let map3 = map1 |> Map.map (fun key value -> value.ToLower())
    printfn "%A" map1
    printfn "%A" map2
    printfn "%A" map3

module ParallelProgramming = 

    let isPrime n = 
        let upperBound = int (sqrt (float n))
        let hasDivisor =     
            [2..upperBound]
            |> List.exists (fun i -> n % i = 0)

        not hasDivisor
        
    let nums = [|1..500000|]
    let finalDigitOfPrimes = 
            nums 
            |> PSeq.filter isPrime
            |> PSeq.groupBy (fun i -> i % 10)
            |> PSeq.map (fun (k, vs) -> (k, Seq.length vs))
            |> PSeq.toArray  

    let averageOfFinalDigit = 
        nums 
        |> PSeq.filter isPrime
        |> PSeq.groupBy (fun i -> i % 10)
        |> PSeq.map (fun (k, vs) -> (k, Seq.length vs))
        |> PSeq.averageBy (fun (k,n) -> float n)

    let sumOfLastDigitsOfPrimes = 
        nums 
        |> PSeq.filter isPrime
        |> PSeq.sumBy (fun x -> x % 10)

    let oneBigArray = [| 0..1000000 |]
    
    // do some CPU intensive computation
    let rec computeSomeFunction x = 
        if x <= 2 then 1
        else computeSomeFunction (x - 1) + computeSomeFunction (x - 2)
    
    //#time
    // Do a parallel map over a large input array
    let computeResults() = oneBigArray |> Array.Parallel.map (fun x -> computeSomeFunction (x % 20))
    
    printfn "Parallel computation results: %A" (computeResults())

    let computeResults'() = oneBigArray |> Array.map (fun x -> computeSomeFunction (x % 20))
    
    printfn "Sync computation results: %A" (computeResults'())