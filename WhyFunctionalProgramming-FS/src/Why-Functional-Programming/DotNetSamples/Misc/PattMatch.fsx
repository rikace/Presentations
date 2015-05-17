
let data = ("Cleveland", 390000)
let city, population = data

let x = 9
match x with
  | num when num < 10 -> printfn "Less than ten"
  | _ -> printfn "Greater than or equal to ten"
