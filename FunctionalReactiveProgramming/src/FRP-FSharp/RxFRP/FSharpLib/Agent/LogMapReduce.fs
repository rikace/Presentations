namespace Easj360FSharp

module LogMapReduce = 

    open System.IO
    open System.Collections.Generic

    let inputFile = @"web.log"
    let mapLogFileIpAddr logFile =
        let fileReader logFile = 
            seq { use fileReader = new StreamReader(File.OpenRead(logFile))
                while not fileReader.EndOfStream do
                    yield fileReader.ReadLine()}    

        //Takes lines and extracts IP Address Out, filter invalid lines out first
        let cutIp = 
            let line = fileReader inputFile 
            line
            |> Seq.filter (fun line -> not (line.StartsWith("#")))
            |> Seq.map (fun line -> line.Split [|' '|])
            |> Seq.map (fun line -> line.[8],1)
            |> Seq.toArray
        cutIp

    let ipMatches = mapLogFileIpAddr inputFile
    let reduceFileIpAddr = 
        Array.fold
            (fun (acc : Map<string, int>) ((ipAddr, num) : string * int) ->
                if Map.containsKey ipAddr acc then
                    let ipFreq = acc.[ipAddr]
                    Map.add ipAddr (ipFreq + num) acc
                else
                    Map.add ipAddr 1 acc)
            Map.empty
            ipMatches

    //Display Top 10 Ip Addresses
    let topIpAddressOutput reduceOutput = 
        let sortedResults = 
            reduceFileIpAddr
            |> Map.toSeq
            |> Seq.sortBy (fun (ip, ipFreq) -> -ipFreq) 
            |> Seq.take 10
        sortedResults
        |> Seq.iter(fun (ip, ipFreq) ->
            printfn "%s, %d" ip ipFreq);;

    //reduceFileIpAddr |> topIpAddressOutput
   // System.Console.Read() |> ignore
