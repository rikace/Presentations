namespace PerfUtil

    open System

    open PerfUtil.Utils

    type TimeComparer (?errorMargin : TimeSpan) =
        let errorMargin = defaultArg errorMargin TimeSpan.Zero
        interface IPerformanceComparer with
            member __.IsBetterOrEquivalent (current : PerfResult) (other : PerfResult) =
                if other.HasFailed then true
                else
                    other.Elapsed - current.Elapsed >= errorMargin

            member __.GetComparisonMessage current other = defaultComparisonMessage current other

    and AbsoluteComparer () =
        interface IPerformanceComparer with
            member __.IsBetterOrEquivalent (current : PerfResult) (other : PerfResult) =
                if other.HasFailed then true
                else
                    current.Elapsed <= other.Elapsed
                        && current.CpuTime <= other.CpuTime 
                        && getSpace current <= getSpace other

            member __.GetComparisonMessage current other = defaultComparisonMessage current other

    /// The mean comparer compares space and time usage for the subject under test
    /// with another PerfResult to see if the current run as better or worse. Also see
    /// TimeComparer and AbsoluteComparer.
    ///
    /// <param name="spaceFactor">Specifies the weight to assign space usage when making
    /// the decision on whether the current run was better or worse. By default the weight
    /// is 0.2 and time is 1 - 0.2. Accepts [0.0, 1.0].</param>
    ///
    /// <param name="leastAcceptableImprovementFactor">
    /// A factor of 'how much better' the this implementation needs to be the other, defaulting
    /// to 1.0, meaning it must be at least as good. Turn this knob down to say that you need
    /// this implementation to be slightly less good, or up to ensure it's better than the
    /// other.</param>
    and WeightedComparer (?spaceFactor : float, ?leastAcceptableImprovementFactor) =
        let spaceFactor = defaultArg spaceFactor 0.2
        let timeFactor = 1. - spaceFactor
        let leastAcceptableImprovementFactor = defaultArg leastAcceptableImprovementFactor 1.

        do if spaceFactor < 0. || spaceFactor > 1. then invalidArg "spaceFactor" "value must be between 0 and 1."

        interface IPerformanceComparer with
            member __.IsBetterOrEquivalent (current : PerfResult) (other : PerfResult) =
                if other.HasFailed then true
                else
                    let dtime = getTimeSpanRatio current.Elapsed other.Elapsed
                    let dspace = getSpaceRatio current other
                    dtime * timeFactor + dspace * spaceFactor >= leastAcceptableImprovementFactor

            member __.GetComparisonMessage current other = defaultComparisonMessage current other