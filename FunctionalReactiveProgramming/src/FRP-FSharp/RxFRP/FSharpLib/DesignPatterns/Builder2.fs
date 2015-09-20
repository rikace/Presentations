module Builder2

// pizza module that defines all recipes
[<AutoOpen>]
module PizzaModule = 
    let makeNormalDough () = printfn "make normal dough"
    let makePanBakedDough () = printfn "make pan baked dough"
    let makeCrossDough() = printfn "make cross dough"

    let makeHotSauce() = printfn "make hot sauce"
    let makeMildSauce() = printfn "make mild sauce"
    let makeLightSauce() = printfn "make light sauce"

    let makePepperoniTopping() = printfn "make pepperoni topping"
    let makeFiveCheeseTopping() = printfn "make five cheese topping"
    let makeBaconHamTopping() = printfn "make bacon ham topping"

// cook takes the recipe and ingredients and makes the pizza

let cook pizza recipeSteps = 
    printfn "making pizza %s" pizza
    recipeSteps 
    |> List.iter(fun f -> f())

[ makeNormalDough; makeMildSauce    
  makePepperoniTopping ]
|> cook "pepperoni pizza"
