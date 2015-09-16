namespace PerfUtil.CaseStudy

    open PerfUtil
    open PerfUtil.NUnit

    open NUnit.Framework

    // create NUnit test beds

    type ``Serializer Comparer``() =
        inherit NUnitPerf<Serializer> ()
        
        let tests = PerfTest.OfModuleMarker<Tests.Marker> ()
        let tester = SerializationPerf.CreateImplementationComparer(throwOnError = true)

        override __.PerfTester = tester :> _
        override __.PerfTests = tests


    type ``Past Version Comparer`` () =
        inherit NUnitPerf<Serializer> ()

        let persistResults = true
        let historyFile = "fspResults.xml"

        let tests = PerfTest.OfModuleMarker<Tests.Marker> ()
        let tester = SerializationPerf.CreatePastVersionComparer(historyFile, throwOnError = true)

        override __.PerfTester = tester :> _
        override __.PerfTests = tests

        [<TestFixtureTearDown>]
        member __.PersistResults () =
            if persistResults then tester.PersistCurrentResults()