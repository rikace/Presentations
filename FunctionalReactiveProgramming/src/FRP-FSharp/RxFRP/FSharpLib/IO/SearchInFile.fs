namespace Easj360FSharp 

open System
open System.IO

module SearchInFile =

    type Search() =
    
        let (|Default|) defaultValue input =
            defaultArg input defaultValue
    
        let rec getFiles dir filter =
            seq {   yield! System.IO.Directory.EnumerateFiles(dir, filter)
                    for subDir in System.IO.Directory.EnumerateDirectories(dir) do
                        yield! getFiles subDir filter }

    
        let readAndSearchInFile (path:string, parse:string) =
           seq {   use reader = new System.IO.StreamReader(path)
                   let counter = ref 0
                   while not reader.EndOfStream do
                        incr counter
                        if (reader.ReadLine()).Contains(parse) then 
                            yield !counter, path }
    
        let SearchInFile folder parse (Default "*.*" filter) =
            let analyze =
                getFiles folder filter
                |> Seq.filter System.IO.File.Exists
                |> Seq.map (fun f -> readAndSearchInFile(f, parse))
                |> Seq.concat
            analyze
   
        let SearchInFileParallel folder parse (Default "*.*" filter) =
            let analyze =
                getFiles folder filter
                |> Seq.filter System.IO.File.Exists
                |> Seq.map (fun f -> async { return readAndSearchInFile(f, parse) })        
                |> Async.Parallel
                |> Async.RunSynchronously
                |> Seq.concat
            analyze

        
        let readAndSearchInFiledAsync (path:string, parse:string) = async { return readAndSearchInFile(path, parse) }

        let agent = MailboxProcessor.Start(fun m ->
            async{
                while true do
                    let! (f,p) = m.Receive()
                    //printfn "%s - %s" f p
                    let! res = Async.StartChild(readAndSearchInFiledAsync(f, p))
                    let! resDone = res
                    resDone            
                    |> Seq.iter (fun (i, x) -> printfn "File %s - Line %d" x i)
                })
        
        let rec traverse dir parse filter =
            async {
                for file in System.IO.Directory.EnumerateFiles(dir, filter)
                    do agent.Post(file, parse)
                return! [for d in System.IO.Directory.EnumerateDirectories(dir)
                            do yield traverse d parse filter]
                            |> Async.Parallel |> Async.Ignore   }