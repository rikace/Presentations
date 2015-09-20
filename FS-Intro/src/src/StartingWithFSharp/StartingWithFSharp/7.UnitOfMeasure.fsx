//  _    _       _ _                __      
// | |  | |     (_) |              / _|     
// | |  | |_ __  _| |_ ___    ___ | |_      
// | |  | | '_ \| | __/ __|  / _ \|  _|     
// | |__| | | | | | |_\__ \ | (_) | |       
//  \____/|_| |_|_|\__|___/  \___/|_| 

//  _ __ ___   ___  __   ___ _   _ _ __ ___
// | '_ ` _ \ / _ \/ _` / __| | | | '__/ _ \
// | | | | | |  __/ (_| \__ \ |_| | | |  __/
// |_| |_| |_|\___|\__,_|___/\__,_|_|  \___|


[<Measure>] type Metre

let x = 10<Metre>


let Add5Metres x =
    x + 5<Metre>

Add5Metres x

Add5Metres 5<Metre>

Add5Metres 5




// ============================
// Units of Measurement
// ============================
module UnitofMeasure = 
    // Units of measure allow you to pass along unit information
    [<Measure>]
    type fahrenheit

    let printTemperature (temp : float<fahrenheit>) =

        if   temp < 32.0<_>  then
            printfn "Below Freezing!"
        elif temp < 65.0<_>  then
            printfn "Cold"
        elif temp < 75.0<_>  then
            printfn "Just right!"
        elif temp < 100.0<_> then
            printfn "Hot!"
        else
            printfn "Scorching!"

    // Because the function only accepts fahrenheit values, 
    // it will fail to work with any floating-point values 
    // encoded with a different unit of measure

    let seattle = 59.0<fahrenheit>

    printTemperature seattle


    // ERROR: Different units
    [<Measure>]
    type celsius

    let nyc = 18.0<celsius>
    
    printTemperature nyc


    // Define a measure for meters
    [<Measure>]
    type m

    // Multiplication, goes to meters squared
    // val it : float<m ^ 2> = 1.0
    1.0<m> * 1.0<m>

    // Division, drops unit entirely
    // val it : float = 1.0
    1.0<m> / 1.0<m>

    // Repeated division, results in 1 / meters
    1.0<m> / 1.0<m> / 1.0<m>

    [<Measure>]
    type mile = 
        static member asMeter = 1600.<m/mile>
    
    let d = 50.<mile> // Distance expressed using imperial units
    let d' = d * mile.asMeter // Same distance expressed using metric system
    
    printfn "%A = %A" d d'

    let error = d + d'       // Compile error: units of measure do not match


    // define some measures
    [<Measure>] 
    type cm

    [<Measure>] 
    type inches

    [<Measure>] 
    type feet =
       // add a conversion function
       static member toInches(feet : float<feet>) : float<inches> = 
          feet * 12.0<inches/feet>

    // define some values
    let meter = 100.0<cm>
    let yard = 3.0<feet>

    //convert to different measure
    let yardInInches = feet.toInches(yard)

    // can't mix and match!
    yard + meter

    // now define some currencies
    // test
    [<Measure>] type EUR
    [<Measure>] type USD
    [<Measure>] type GBP

    let gbp10 = 10.0<GBP>
    let usd10 = 10.0<USD>

    gbp10 + gbp10             // allowed: same currency
    gbp10 + usd10             // not allowed: different currency
    gbp10 + 1.0               // not allowed: didn't specify a currency
    gbp10 + 1.0<_>            // allowed using wildcard


    type CurrencyRate<[<Measure>]'u, [<Measure>]'v> = 
        { Rate: float<'u/'v>; Date: System.DateTime}


    let mar1 = System.DateTime(2012,3,1)
    let eurToUsdOnMar1 = {Rate= 1.2<USD/EUR>; Date=mar1 }
    let eurToGbpOnMar1 = {Rate= 0.8<GBP/EUR>; Date=mar1 }

    let tenEur = 10.0<EUR>
    let tenEurInUsd = eurToUsdOnMar1.Rate * tenEur 




    [<Measure>] type foot
    [<Measure>] type ft = foot
    [<Measure>] type sqft = foot ^ 2
    [<Measure>] type meter
    //[<Measure>] type m = meter
    [<Measure>] type mSqrd = m ^ 2

    // convert from unitless to units
    let measuredInFeet = 15.0 * 1.0<ft>
    let unitLess = 15.0<ft> / 1.0<ft>
    let squarefeet = 15.0<foot> * 16.0<foot> // note the return type ends up being <ft ^ 2> even though I defined sqft

    let getArea (b : int<ft>) (h : int<ft>) : int<sqft> = b * h

    // let areaInMismatch = getArea 15<m> 16<m>

    let areaInSqft = getArea 15<ft> 16<ft>

    let getArea' (b : int<'u>) (h : int<'u>) : int<'u ^ 2> = b * h

    let areaInSomeUnits = getArea' 15<m> 16<m>

    [<Measure>] type inch =
                    static member perFoot = 12.0<inch/foot>

    let inchesPerFoot = 3.0<foot> * inch.perFoot
    

    [<Measure>] type farenheit
    [<Measure>] type celsius

    let fromFarenheitToCelsius (f : float<farenheit>) = ((float f - 32.0) * (5.0/9.0)) * 1.0<celsius>
    let fromCelsiusToFarenheit (c : float<celsius>) = ((float c * (9.0/5.0)) + 32.0) * 1.0<farenheit>
    
    // Type extensions 
    type farenheit with static member toCelsius = fromFarenheitToCelsius
                        static member fromCelsius = fromCelsiusToFarenheit
    
    type celsius with   static member toFarenheit = fromCelsiusToFarenheit
                        static member fromFarenheit = fromFarenheitToCelsius

    let printFToC () =
        let brrr = farenheit.toCelsius 35.0<farenheit>
        printfn "%A" brrr

    let printCToF () =
        let brrr = celsius.toFarenheit 4.0<celsius>
        printfn "%A" brrr

