module UnitOfMeasure

[<Measure>] 
type mi

[<Measure>] 
type km

// define some values
let mike = [| 6<mi>; 9<mi>; 5<mi>; 18<mi> |]
let chris = [| 3<km>; 5<km>; 2<km>; 8<km> |]

let totalDistance = (Array.append mike chris) |> Array.sum


