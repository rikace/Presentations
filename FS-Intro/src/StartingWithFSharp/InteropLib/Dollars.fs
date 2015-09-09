namespace InteropLib

[<CLIMutableAttribute>]
type DollarsFS = 
    { Amount : decimal }
    member x.Times(multiplier) = { Amount = x.Amount * multiplier }

//let d1 = { Amount = 10M }
//let d2 = { Amount = 5M }
//
//d1 = d2.Times(2M)
