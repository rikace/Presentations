// Strategy

// quick sort algorithm
let quicksort l = 
    printfn "quick sort"

// shell short algorithm
let shellsort l = 
    printfn "shell short"

// bubble short algorithm
let bubblesort l = 
    printfn "bubble sort"

// define the strategy class
type Strategy() = 
    let mutable sortFunction = fun _ -> ()
    member this.SetStrategy f = sortFunction <- f
    member this.Execute n = sortFunction n

let strategy() = 
    let s = Strategy()

    // set strategy to be quick sort
    s.SetStrategy quicksort
    s.Execute [1..6]

    // set strategy to be bubble sort
    s.SetStrategy bubblesort
    s.Execute [1..6]

strategy()