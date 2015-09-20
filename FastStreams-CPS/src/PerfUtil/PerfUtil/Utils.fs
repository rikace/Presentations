namespace PerfUtil

    open System

    #nowarn "42"

    module internal Utils =

        let currentHost = System.Net.Dns.GetHostName()

        [<Literal>]
        let gcGenWeight = 10
        
        // computes a performance improvement factor out of two given timespans
        // add + 1L to eliminate the slim possibility of division by zero's and NaN's
        // ticks register large numbers so this shouldn't skew the final result significantly.
        let getTimeSpanRatio (this : TimeSpan) (that : TimeSpan) =
            float (decimal (that.Ticks + 1L) / decimal (this.Ticks + 1L))

        // computes a polynomial value out of gc garbage collection data
        // [ gen0 ; gen1 ; gen2 ] -> gen0 + 10 * gen1 + 10^2 * gen2
        let getSpace (r : PerfResult) = 
            r.GcDelta
            |> List.mapi (fun i g -> g * pown gcGenWeight i) 
            |> List.sum

        let getSpaceRatio (this : PerfResult) (that : PerfResult) =
            (float (getSpace that) + 0.1) / (float (getSpace this) + 0.1)

        // add single quotes if text contains whitespace
        let quoteText (text : string) =
            if text |> Seq.exists Char.IsWhiteSpace then
                sprintf "'%s'" text
            else
                text

        let defaultComparisonMessage (this : PerfResult) (other : PerfResult) =
            assert(this.TestId = other.TestId)

            match this.Error, other.Error with
            | Some e, _ ->
                sprintf "%s: %s failed with %O." this.TestId (quoteText this.SessionId) e
            | _, Some e ->
                sprintf "%s: %s failed with '%s'." other.TestId (quoteText other.SessionId) e
            | _ ->
                sprintf "%s: %s was %.2fx faster and %.2fx more memory efficient than %s."
                    (quoteText other.TestId)
                    (quoteText this.SessionId)
                    (getTimeSpanRatio this.Elapsed other.Elapsed)
                    (getSpaceRatio this other)
                    (quoteText other.SessionId)

        /// only returns items in enumeration that appear as duplicates
        let getDuplicates xs =
            xs
            |> Seq.groupBy id 
            |> Seq.choose (fun (id, vs) -> if Seq.length vs > 1 then Some id else None)


        //
        //  PerfTest activation code
        //

        open System.Reflection

        type MemberInfo with
            member m.TryGetAttribute<'T when 'T :> Attribute> () =
                match m.GetCustomAttributes(typeof<'T>, true) with
                | [||] -> None
                | attrs -> attrs.[0] :?> 'T |> Some

        type Delegate with
            static member Create<'T when 'T :> Delegate> (parentObj : obj, m : MethodInfo) =
                if m.IsStatic then
                    Delegate.CreateDelegate(typeof<'T>, m) :?> 'T
                else
                    Delegate.CreateDelegate(typeof<'T>, parentObj, m) :?> 'T


        type SynVoid =
            static member Swap(t : Type) =
                if t = typeof<Void> then typeof<SynVoid> else t

        type MethodWrapper =
            static member WrapUntyped<'Param> (parentObj : obj, m : MethodInfo) =
                typeof<MethodWrapper>
                    .GetMethod("Wrap", BindingFlags.NonPublic ||| BindingFlags.Static)
                    .MakeGenericMethod([| typeof<'Param> ; SynVoid.Swap m.ReturnType |])
                    .Invoke(null, [| parentObj ; box m |]) :?> 'Param -> unit

            static member Wrap<'Param, 'ReturnType> (parentObj, m : MethodInfo) =
                if typeof<'ReturnType> = typeof<SynVoid> then
                    let d = Delegate.Create<Action<'Param>>(parentObj, m)
                    d.Invoke
                else
                    let d = Delegate.Create<Func<'Param, 'ReturnType>> (parentObj, m)
                    d.Invoke >> ignore

        let getPerfTestsOfType<'Impl when 'Impl :> ITestable> ignoreAbstracts bindingFlags (t : Type) =
            let defBindings = BindingFlags.Public ||| BindingFlags.Static ||| BindingFlags.Instance
            let bindingFlags = defaultArg bindingFlags defBindings

            if t.IsGenericTypeDefinition then
                failwithf "Container type '%O' is generic." t

            let tryGetPerfTestAttr (m : MethodInfo) =
                match m.TryGetAttribute<PerfTestAttribute> () with
                | Some attr ->
                    if m.IsGenericMethodDefinition then
                        failwithf "Method '%O' marked with [<PerfTest>] attribute but is generic." m

                    match m.GetParameters() |> Array.map (fun p -> p.ParameterType) with
                    | [| param |] when param = typeof<'Impl> -> Some(m, attr)
                    | [| param |] when typeof<ITestable>.IsAssignableFrom(param) -> None
                    | _ -> 
                        failwithf "Method '%O' marked with [<PerfTest>] attribute but contains invalid parameters." m
                | None -> None

            let perfMethods = t.GetMethods(bindingFlags) |> Array.choose tryGetPerfTestAttr

            let requireInstance = perfMethods |> Array.exists (fun (m,_) -> not m.IsStatic)

            if ignoreAbstracts && t.IsAbstract && requireInstance then []
            else
                let instance = 
                    if requireInstance then
                        Activator.CreateInstance t
                    else
                        null

                let perfTestOfMethod (m : MethodInfo, attr : PerfTestAttribute) =
                    {
                        Id = sprintf "%s.%s" m.DeclaringType.Name m.Name
                        Repeat = attr.Repeat
                        Test = MethodWrapper.WrapUntyped<'Impl> (instance, m) 
                    }

                perfMethods |> Seq.map perfTestOfMethod |> Seq.toList