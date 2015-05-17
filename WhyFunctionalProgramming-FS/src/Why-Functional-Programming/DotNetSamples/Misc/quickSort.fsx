let rec quicksort list =
   match list with
   | [] ->                            // If the list is empty
        []                            // return an empty list
   | firstElem::otherElements ->      // If the list is not empty     
        let smallerElements =         // extract the smaller ones    
            otherElements             
            |> List.filter (fun e -> e < firstElem) 
            |> quicksort              // and sort them
        let largerElements =          // extract the large ones
            otherElements 
            |> List.filter (fun e -> e >= firstElem)
            |> quicksort              // and sort them
        // Combine the 3 parts into a new list and return it
        List.concat [smallerElements; [firstElem]; largerElements]

//test
printfn "%A" (quicksort [1;5;23;18;9;1;3])


let rec quicksort2 = function
   | [] -> []                         
   | first::rest -> 
        let smaller,larger = List.partition ((>=) first) rest 
        List.concat [quicksort2 smaller; [first]; quicksort2 larger]
        
// test code        
printfn "%A" (quicksort2 [1;5;23;18;9;1;3])
