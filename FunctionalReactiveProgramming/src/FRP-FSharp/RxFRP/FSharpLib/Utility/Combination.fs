namespace Easj360FSharp 

open System
open System.Numerics // BigInteger class

module ModuleCombination =
 
    type Combination(n : int, k : int, a : int[]) = // better in F# to make the ctor with most args the primary ctor
      do if n < 0 || k < 0 then failwith "Negative argument in Combination ctor"
      do if n < k then failwith "Subset size k is larger than n in Combination ctor"
      let n : int = n // private
      let k : int = k
      let data = [| for i = 0 to a.Length-1 do yield a.[i] |] // copy input array to private data array ("->" token no work here)

      new(n : int, k : int) = // secondary constructor, makes an initial n,k Combination
        do if n < 0 || k < 0 then failwith "Negative argument in Combination ctor"
        do if n < k then failwith "Subset size k is larger than n in Combination ctor"
        let starters = [| for i in 0..k-1 -> i |] // -> can be used instead of do-yield keywords here
        new Combination(n,k,starters) 
 
      //member this.Data with get() = data
      //member this.Data with set(value) = data <- value

      member this.IsLast() : bool =
        if data.[0] = n-k then true
        else false
 
      member this.Successor() : Combination =
        let temp = [| for i in 0..k-1 -> data.[i] |] // copy input to temp array
        let mutable x = k-1 // find "x" - right-most index to change
        while x > 0 && temp.[x] = n - k + x do 
          x <- x - 1
        temp.[x] <- temp.[x] + 1 // increment value at x
        for j = x to k-2 do // increment all values to the right of x
          temp.[j+1] <- temp.[j] + 1
    
        let result = new Combination(n, k, temp) // use secondary ctor 
        result

      member this.ApplyTo(a : string[]) : string[] =
        if a.Length <> n then failwith "Invalid array size in Combination.ApplyTo()"
        let result = Array.zeroCreate k
        for i = 0 to k-1 do
          result.[i] <- a.[data.[i]]
        result

      override this.ToString() : string =
        let mutable s : string = "^ "
        for i in 0..k-1 do
          s <- s + data.[i].ToString() + " "
        s <- s + "^"
        s

      static member Choose(n : int, k : int) : BigInteger =
        if n < 0 || k < 0 then failwith "Negative argument passed to Combination.Choose()"
        if n < k then failwith "Subset size k is larger than n in Combination.Choose()"
        let (delta, iMax) =
          if k < n-k then
            (n-k, k)
          else
            (k, n-k)
        let mutable answer : BigInteger = bigint delta + bigint 1 // bigint() is a function that maps to the BigInteger structure ctor
        for i = 2 to iMax do
          answer <- (answer * (bigint delta + bigint i )) / bigint i
        answer

      static member Factorial(n : int) : BigInteger =
        if n < 0 then failwith "Negative argument passed to Combination.Factorial()"
        let mutable answer : BigInteger = bigint 1
        for i in 1..n do
          answer <- answer * bigint i
        answer
  
    // end type Combination

    type Permutation(n : int, a : int[]) =
      do if n < 0 then failwith "Negative argument in Permutation ctor"
      let n = n
      let data = [| for i = 0 to a.Length-1 do yield a.[i] |]

      new(n : int) =
        if n < 0 then failwith "Negative argument in Permutation ctor"
        let starters = [| for i in 0..n-1 -> i |]
        new Permutation(n, starters)
  
      override this.ToString() : string =
        let mutable s : string = "% "
        for i in 0..n-1 do
          s <- s + data.[i].ToString() + " "
        s <- s + "%"
        s

      member this.Successor() : Permutation =
        let temp = [| for i in 0..n-1 -> data.[i] |] // copy current data to temp array
        let mutable left = n-2
        while temp.[left] > temp.[left+1] && left >= 1 do
          left <- left - 1
        let mutable right = n - 1
        while temp.[left] > temp.[right] do
          right <- right - 1
        //(temp.[left], temp.[right]) <- (temp.[right], temp.[left])
        let t = temp.[left]
        temp.[left] <- temp.[right]
        temp.[right] <- t 
    
        let mutable i = left + 1
        let mutable j = n - 1 
        while i < j do
          //(temp.[i], temp.[j]) = (temp.[j], temp.[i])
          let t = temp.[i]
          temp.[i] <- temp.[j]
          temp.[j] <- t
          i <- i + 1
          j <- j - 1
        let result = new Permutation(n, temp) // make Permutation from temp array
        result

      member this.ApplyTo(a : string[]) : string[] =
        if a.Length <> n then failwith "Invalid array size in Permutation.ApplyTo()"
        let result = Array.zeroCreate n
        for i = 0 to n-1 do
          result.[i] <- a.[data.[i]]
        result

      member this.IsLast() : bool =
        let mutable i = 0
        while i <= n-1 && data.[i] = n-i-1 do
          i <- i + 1
        i = n // if i = n, means all data passed the check for last element

    // end type Permutation

    type UsageCombination() =
        member x.Start() =
            try
              printfn "\nBegin combinations and permutations with F# demo\n"
              printfn "All combinations of 5 items taken 3 at a time in lexicographical order are: \n"
              let mutable c = new Combination(5,3)
              printfn "%A" c // print initial combination
              while c.IsLast() = false do // objects cannot be null in F# so use an explicit method
                c <- c.Successor()
                printfn "%A" c
   
              printf "\nThe last combination applied to array [| \"ant\"; \"bat\"; \"cow\"; \"dog\"; \"elk\" |] is: \n"
              let animals = [| "ant"; "bat"; "cow"; "dog"; "elk" |]
              //let result =  c.ApplyTo(animals)
              let result = animals |> c.ApplyTo
              printfn "%A" result

              printfn "\nThe number of ways to Choose 200 items 10 at a time = Choose(200,10) ="
              let Choose_200_10 = Combination.Choose(200,10).ToString("000,000")
              printfn "%s" Choose_200_10

              printfn "\nThe number of ways to arrange 52 cards = 52! = "
              let Factorial_52 = Combination.Factorial(52).ToString("000,000")
              printfn "%s" Factorial_52

              printfn "\nAll permutations of 3 items in lexicographical order are: \n"
              let mutable p = new Permutation(3)
              printfn "%A" p // print initial permutation
              while p.IsLast() = false do
                p <- p.Successor() 
                printfn "%A" p
      
              printfn "\nEnd demo\n"
              Console.ReadLine() |> ignore

            with
              | Failure(errorMsg) -> printfn "Fatal error: %s" errorMsg

            // end program