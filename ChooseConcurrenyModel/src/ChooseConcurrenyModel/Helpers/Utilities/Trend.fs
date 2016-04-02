namespace  Utilities
open System.Diagnostics.CodeAnalysis

/// Linear trend from slope and intercept. Predict y given any x value 
/// using the formula y = slope * x + intercept.
[<SuppressMessage("Microsoft.Performance", "CA1815:OverrideEqualsAndOperatorEqualsOnValueTypes")>] 
[<Struct>]
type Trend(slope:float, intercept:float) = 
    /// The change in y per unit of x.
    member x.Slope = slope

    /// The value of y when x is zero.
    member x.Intercept = intercept

    /// Predicts a y value given any x value using the formula y = slope * x + intercept.
    member x.Predict(ordinate) =
        slope * ordinate + intercept
