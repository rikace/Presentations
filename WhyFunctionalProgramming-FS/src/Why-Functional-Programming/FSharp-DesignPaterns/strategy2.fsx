// quick sort algorithm
let quicksort l = 
    printfn "quick sort"

// shell short algorithm
let shellsort l = 
    printfn "shell short"

// bubble short algorithm
let bubblesort l = 
    printfn "bubble sort"

let executeStrategy f n = f n 

let strategy() = 
    // set strategy to be quick sort
    let s = executeStrategy quicksort
    // execute the strategy against a list of integers
    [1..6] |> s

    // set strategy to be bubble sort
    let s2 = executeStrategy bubblesort
    // execute the strategy against a list of integers
    [1..6] |> s2

strategy()