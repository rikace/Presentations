namespace PerfUtil

    open PerfUtil.Persist
    open PerfUtil.Utils

    open System
    open System.Reflection

    [<AutoOpen>]
    module Extensions =

        let inline repeat times (testF : 'State -> unit) (state : 'State) =
            for i = 1 to times do testF state

        type PerfTest<'Impl when 'Impl :> ITestable> with
            
            /// load all methods containing the [<PerfTest>] attribute from given type.
            static member OfType(container : Type, ?bindingFlags) =
                getPerfTestsOfType<'Impl> false bindingFlags container

            /// load all methods containing the [<PerfTest>] attribute from given type.
            static member OfType<'Container>(?bindingFlags) = 
                getPerfTestsOfType<'Impl> false bindingFlags typeof<'Container>

            /// load all methods containing the [<PerfTest>] attribute from module marker.
            static member OfModuleMarker<'Marker>(?bindingFlags) =
                getPerfTestsOfType<'Impl> false bindingFlags typeof<'Marker>.DeclaringType

            /// load all methods containing the [<PerfTest>] attribute from assembly.
            static member OfAssembly(a : Assembly, ?bindingFlags) =
                a.GetTypes() 
                |> Seq.filter (fun t -> not t.IsGenericTypeDefinition)
                |> Seq.collect (getPerfTestsOfType<'Impl> true bindingFlags) 
                |> Seq.toList


        [<RequireQualifiedAccess>]
        module PerfTest =

            /// initializes a new performance tester and executes given tests
            let run<'Impl when 'Impl :> ITestable> 
                (testerFactory : unit -> PerformanceTester<'Impl>) (perfTests : PerfTest<'Impl> list) =

                let tester = testerFactory ()

                for test in perfTests do
                    tester.RunTest test

                tester.GetTestResults ()
            

        [<RequireQualifiedAccess>]
        module TestSession =

            /// takes a collection of test sessions and groups them by matching test Id's.
            let groupByTest (tests : TestSession list) =
                tests 
                |> Seq.collect (fun t -> t.Results |> Map.toSeq |> Seq.map snd)
                |> Seq.groupBy (fun br -> br.TestId)
                |> Seq.map (fun (k,vs) -> (k, Seq.toList vs))
                |> Map.ofSeq

            /// persist a collection of test sessions to a given file path.
            let toFile (path : string) (tests : TestSession list) =
                sessionsToFile "" path tests

            /// load a collection of test sessions from xml file.
            let ofFile (path : string) = 
                match sessionsOfFile path with
                | None -> raise <| new System.IO.FileNotFoundException()
                | Some(_,sessions) -> sessions