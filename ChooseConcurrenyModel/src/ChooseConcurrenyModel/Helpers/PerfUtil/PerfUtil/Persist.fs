namespace PerfUtil

    module internal Persist =

        open System
        open System.IO
        open System.Xml.Linq

        let private xn name = XName.Get name

        let testToXml (br : PerfResult) =

            XElement(xn "testResult",
                XAttribute(xn "testId", br.TestId),
                XAttribute(xn "testDate", br.Date),
                XElement(xn "elapsedTime", br.Elapsed.Ticks),
                XElement(xn "cpuTime", br.CpuTime.Ticks),
                XElement(xn "repeat", br.Repeat),
                seq { 
                    match br.Error with 
                    | Some msg -> yield XElement(xn "error", msg) 
                    | None -> ()
                },
                XElement(xn "gcDelta",
                    br.GcDelta 
                    |> List.mapi (fun gen delta -> XElement(xn <| sprintf "gen%d" gen, delta)))
                )

        let testOfXml sessionId (xEl : XElement) =
            {
                TestId = xEl.Attribute(xn "testId").Value
                SessionId = sessionId
                Date = xEl.Attribute(xn "testDate").Value |> DateTime.Parse

                Error = 
                    match xEl.Element(xn "error") with
                    | null -> None
                    | xel ->
                        if String.IsNullOrWhiteSpace xel.Value then None
                        else Some xel.Value

                Elapsed = xEl.Element(xn "elapsedTime").Value |> int64 |> TimeSpan.FromTicks
                CpuTime = xEl.Element(xn "cpuTime").Value |> int64 |> TimeSpan.FromTicks
                Repeat = 
                    match xEl.Element(xn "repeat") with
                    | null -> 1
                    | xel ->
                        let ok,v = Int32.TryParse xel.Value
                        if ok then v
                        else 1

                GcDelta =
                    xEl.Element(xn "gcDelta").Elements()
                    |> Seq.map (fun gc -> int gc.Value)
                    |> Seq.toList
            }

        let sessionToXml (tests : TestSession) =
            XElement(xn "testRun",
                XAttribute(xn "id", tests.Id),
                XAttribute(xn "hostname", tests.Hostname),
                XAttribute(xn "date", tests.Date),
                tests.Results 
                |> Map.toSeq 
                |> Seq.map snd 
                |> Seq.sortBy (fun b -> b.Date) 
                |> Seq.map testToXml)

        let sessionOfXml (xEl : XElement) =
            let id = xEl.Attribute(xn "id").Value
            let host = xEl.Attributes(xn "hostname") |> Seq.tryPick (fun a -> Some a.Value)
            let date = xEl.Attribute(xn "date").Value |> DateTime.Parse
            let tests = 
                xEl.Elements(xn "testResult") 
                |> Seq.map (testOfXml id) 
                |> Seq.map (fun tr -> tr.TestId, tr)
                |> Map.ofSeq
            {
                Id = id
                Hostname = defaultArg host "unknown"
                Date = date
                Results = tests
            }

        let sessionsToXml (sessionName : string) (session : TestSession list) =
            XDocument(
                XElement(xn "testSuite",
                    XAttribute(xn "suiteName", sessionName),
                    session |> Seq.sortBy(fun s -> s.Date) |> Seq.map sessionToXml))

        let sessionsOfXml (root : XDocument) =
            let xEl = root.Element(xn "testSuite")
            let name = xEl.Attribute(xn "suiteName").Value
            let sessions =
                xEl.Elements(xn "testRun")
                |> Seq.map sessionOfXml
                |> Seq.toList

            name, sessions


        let sessionsToFile name (path : string) (sessions : TestSession list) =
            let doc = sessionsToXml name sessions
            doc.Save(path)
            
        let sessionsOfFile (path : string) =
            let path = Path.GetFullPath path
            if File.Exists(path) then
                XDocument.Load(path) |> sessionsOfXml |> Some
            elif not <| Directory.Exists(Path.GetDirectoryName(path)) then
                invalidOp <| sprintf "invalid path '%s'." path
            else
                None