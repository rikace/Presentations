module Builder

// pizza interface
type IPizza = 
    abstract Name : string with get
    abstract MakeDough : unit->unit
    abstract MakeSauce : unit->unit
    abstract MakeTopping: unit->unit

// pizza module that defines all recipes
[<AutoOpen>]
module PizzaModule = 
    let makeNormalDough() = printfn "make normal dough"
    let makePanBakedDough() = printfn "make pan baked dough"
    let makeCrossDough() = printfn "make cross dough"

    let makeHotSauce() = printfn "make hot sauce"
    let makeMildSauce() = printfn "make mild sauce"
    let makeLightSauce() = printfn "make light sauce"

    let makePepperoniTopping() = printfn "make pepperoni topping"
    let makeFiveCheeseTopping() = printfn "make five cheese topping"
    let makeBaconHamTopping() = printfn "make bacon ham topping"

// define a pepperoni pizza recipe
let pepperoniPizza = 
        {   new IPizza with 
                member this.Name = "Pepperoni Pizza"
                member this.MakeDough() = makeNormalDough()
                member this.MakeSauce() = makeHotSauce()
                member this.MakeTopping() = makePepperoniTopping() }

// cook takes pizza recipe and makes the pizza
let cook(pizza:IPizza) = 
    printfn "making pizza %s" pizza.Name
    pizza.MakeDough()
    pizza.MakeSauce()
    pizza.MakeTopping()

// cook pepperoni pizza
cook pepperoniPizza
