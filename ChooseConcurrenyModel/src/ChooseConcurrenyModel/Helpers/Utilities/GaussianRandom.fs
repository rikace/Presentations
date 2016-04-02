namespace  Utilities

open System

/// Creates a new instance of a normally distributed random value generator
/// using the specified mean and standard deviation and seed.
type GaussianRandom(mean, standardDeviation, ?seed) =
    let random = 
        match seed with 
        | None -> new Random()
        | Some(seed) -> new Random(seed)

        
    /// Samples the distribution and returns a random integer. Returns 
    /// a normally distributed random number rounded to the nearest integer
    member x.NextInteger() =
        int (Math.Floor(x.Next() + 0.5))

    /// Samples the distribution; returns a random sample from a normal distribution
    member x.Next() =
        let mutable x = 0.0

        // Get the next value in the interval (0, 1) 
        // from the underlying uniform distribution
        while x = 0.0 || x = 1.0 do
            x <- random.NextDouble()

        // Transform uniform into normal
        SampleUtilities.GaussianInverse x mean standardDeviation
