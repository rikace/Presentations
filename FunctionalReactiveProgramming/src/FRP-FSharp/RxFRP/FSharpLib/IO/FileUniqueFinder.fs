module FileUniqueFinder
              
open System
open System.IO
open System.Security.Cryptography
open Microsoft.FSharp.Collections
open Microsoft.FSharp.Collections.Array.Parallel
open Microsoft.FSharp.Collections
#if INTERACTIVE
#r "FSharp.powerPack.dll"
#r "FSharp.powerPack.Parallel.Seq.dll"
#endif
//open Microsoft.FSharp.Collections.PSeq
//mp3
//mp4
//wma
let printToConasole = MailboxProcessor.Start(fun agent -> 
                            let rec loop n = async {
                                let! msg = agent.Receive()
                                printfn "%s" msg
                                return! loop (n + 1) }
                            loop 0)


let getUniqueFilesParallel (dirs:string list) filter =
    let rec getFiles dir filter' = seq {    for file in Directory.EnumerateFiles(dir, "*.*", SearchOption.TopDirectoryOnly) do if filter' file then yield FileInfo(file)
                                            for d in Directory.EnumerateDirectories(dir, "*.*", SearchOption.TopDirectoryOnly) do yield! getFiles d filter' }

    let groupPartition f source =  
          source    |> Seq.groupBy f
                    |> Seq.toArray
                    |> Array.Parallel.partition (fun (_, col) -> (Seq.length col) = 1)
             
    let (uniqueFiles, possibleDuplicateFiles) = 
                     dirs |> List.toArray |> Array.Parallel.map (fun dir -> getFiles dir filter) |> Seq.concat
                     |> groupPartition (fun f -> (f.Length, f.Extension))

    let minSizeFile = possibleDuplicateFiles |> Array.Parallel.map (fun (key, _) -> fst key) |> Array.min |> int
   
    let readBytesOffSet = (fun(file:FileInfo) ->
                                use stream = file.OpenRead()
                                let buff = Array.zeroCreate<byte> (int(minSizeFile))
                                let bytes = stream.Read(buff, 0, buff.Length)                                                               
                                printToConasole.Post (sprintf "Computing Bytes file %s" file.Name)
                                (buff.[0..bytes-1] |> Convert.ToBase64String))

    let readMD5Hash = (fun(file:FileInfo) ->
                                use stream = file.OpenRead()
                                use hash = System.Security.Cryptography.MD5.Create()
                                let bytes = hash.ComputeHash(stream)
                                printToConasole.Post (sprintf "Computing Hash file %s" file.Name)
                                (bytes |> Convert.ToBase64String))

    let readBytesOffSetAsync = (fun(file:FileInfo) -> 
                                let op = async {
                                    use stream = file.OpenRead()
                                    let! buff = stream.AsyncRead(int(minSizeFile))
                                    do printToConasole.Post (sprintf "Computing Bytes file %s" file.Name)
                                    return (buff |> Convert.ToBase64String) }
                                Async.RunSynchronously op)

    let hashAsync bufferSize (hashFunction: HashAlgorithm) (stream: Stream) =
        let rec hashBlock currentBlock count = async {
            let buffer = Array.zeroCreate<byte> bufferSize
            let! readCount = stream.AsyncRead buffer
            if readCount = 0 then
                hashFunction.TransformFinalBlock(currentBlock, 0, count) |> ignore
            else 
                hashFunction.TransformBlock(currentBlock, 0, count, currentBlock, 0) |> ignore
                return! hashBlock buffer readCount
        }
        async {
            let buffer = Array.zeroCreate<byte> bufferSize
            let! readCount = stream.AsyncRead buffer
            do! hashBlock buffer readCount
            return hashFunction.Hash |> Convert.ToBase64String
        }
    let readMD5HashAsync = (fun(file:FileInfo) ->
                                let op = async {
                                    use stream = file.OpenRead()
                                    use hash = new SHA256Managed()                                    
                                    printToConasole.Post (sprintf "Computing Hash file %s" file.Name)
                                    let! hashComputed = hashAsync (int stream.Length) hash stream
                                    return hashComputed }
                                Async.RunSynchronously op)

    let arrayMapToList f s = s|> Array.Parallel.map f |> Array.toList

    let rec funcRec (accUnique:FileInfo list, files:seq<FileInfo> list) (fs:(FileInfo -> 'a) list when 'a : equality) =
        match fs with
        | [] -> (accUnique @ (files |> List.map (fun f -> Seq.head f)))
        | (x::xs) ->    let (unique, duplicate) = 
                            files
                            |> Seq.concat
                            |> groupPartition x
                        funcRec ((unique |> arrayMapToList (fun f -> Seq.head (snd f))) @ accUnique, duplicate |> arrayMapToList snd) xs
    funcRec ((uniqueFiles |> arrayMapToList (fun f -> Seq.head (snd f))), (possibleDuplicateFiles |> arrayMapToList (fun l -> snd l))) [readBytesOffSetAsync; readMD5HashAsync] 


let getUniqueFiles (dirs:string list) filter =
    let rec getFiles dir filter' = seq {    for file in Directory.EnumerateFiles(dir, "*.*", SearchOption.TopDirectoryOnly) do if filter' file then yield FileInfo(file)
                                            for d in Directory.EnumerateDirectories(dir, "*.*", SearchOption.TopDirectoryOnly) do yield! getFiles d filter' }

    let groupPartition f source =  
          source    |> Seq.groupBy f
                    |> Seq.toList
                    |> List.partition (fun (_, col) -> (Seq.length col) = 1)
             
    let (uniqueFiles, possibleDuplicateFiles) = 
                     dirs   |> List.map (fun dir -> getFiles dir filter) 
                            |> Seq.concat
                            |> groupPartition (fun f -> (f.Length, f.Extension))

    let minSizeFile = possibleDuplicateFiles |> List.map (fun (key, _) -> fst key) |> List.min |> int
   
    let readBytesOffSet = (fun(file:FileInfo) ->
                                use stream = file.OpenRead()
                                let buff = Array.zeroCreate<byte> (int(minSizeFile))
                                let bytes = stream.Read(buff, 0, buff.Length)                                                               
                                printfn "Computing Bytes file %s" file.Name
                                (buff.[0..bytes-1] |> Convert.ToBase64String))

    let readMD5Hash = (fun(file:FileInfo) ->
                                use stream = file.OpenRead()
                                use hash = System.Security.Cryptography.MD5.Create()
                                let bytes = hash.ComputeHash(stream)
                                printfn "Computing Hash file %s" file.Name
                                (bytes |> Convert.ToBase64String))

    let rec funcRec (accUnique:FileInfo list, files:seq<FileInfo> list) (fs:(FileInfo -> 'a) list when 'a : equality) =
        match fs with
        | [] -> (accUnique @ (files |> List.map (fun f -> Seq.head f)))
        | (x::xs) ->    let (unique, duplicate) = 
                            files
                            |> Seq.concat
                            |> groupPartition x
                        funcRec ((unique |> List.map (fun f -> Seq.head (snd f))) @ accUnique, duplicate |> List.map snd) xs
    funcRec ((uniqueFiles |> List.map (fun f -> Seq.head (snd f))), (possibleDuplicateFiles |> List.map (fun l -> snd l))) [readBytesOffSet; readMD5Hash] 
    

//let dirs = [@"R:\Test\Output"; @"R:\Test\Output1"; @"R:\Test\Output2";@"R:\Test\Output3"; @"R:\Test\Output5"; @"R:\Test\Output4"]
//
//let uniqueFiles = getUniqueFiles dirs (fun s -> true)
//let uniqueFiles' = getUniqueFilesParallel dirs (fun s -> true)
//
//let arr = [1..100]


//let files = dirs
//            |> List.map(fun dir -> getFiles dir (fun _ -> true))
//            |> Seq.concat

//let (a', b') = partitionFiles (fun f -> (f.Length, f.Extension)) files
//let (a'', b'') = b' |> Seq.concat |> partitionFiles (fun f -> readBytesOffSet f)
//let (a''', b''') = b'' |> Seq.concat |> partitionFiles (fun f -> readMD5Hash f)
//
//let getFirst col = col |> Seq.map (fun s -> Seq.exactlyOne s)
//
//[a';a''; a'''; (b''' |> Seq.map |> Seq.head)] 





//[<EntryPointAttribute>]
//let main(args) =
//
//    Console.Write("Press ENTER to Start")
//    Console.ReadLine() |> ignore
//    Console.WriteLine("Starting...")
//    //let dirs = [@"\\GOFLEX_HOME\GoFlex Home Public\Immagini"]
//    
//    //let dirs = [@"R:\Test\Output"; @"R:\Test\Output1"; @"R:\Test\Output2";@"R:\Test\Output3"; @"R:\Test\Output5"; @"R:\Test\Output4"]
//    let dirs = [@"D:\Media\Immagini"; @"F:\Media\Immagini"; @"\\GOFLEX_HOME\GoFlex Home Public\Immagini"]
//
//    let uniqueFiles = getUniqueFilesParallel dirs (fun s -> // s.ToLower().EndsWith(".txt"))
//                                                                s.ToLower().EndsWith(".jpg") ||
//                                                                s.ToLower().EndsWith(".jpeg") ||
//                                                                s.ToLower().EndsWith(".bmp") ||
//                                                                s.ToLower().EndsWith(".avi") ||
//                                                                s.ToLower().EndsWith(".giff"))
//                   
//    let dest = "IMMAGINI"
//    uniqueFiles
//    |> List.iter(fun file -> 
//                        try
//                                let index = file.Directory.FullName.ToUpper().IndexOf(dest)
//                                if(index > 0) then
//                                    let dirDestination = file.Directory.FullName.Replace(file.Directory.Root.Name, @"h:\");
//                                    let destinationFile = Path.Combine(dirDestination, file.Name)
//                                    try
//                                        if not (Directory.Exists dirDestination) then
//                                            ignore( Directory.CreateDirectory dirDestination )
//                                        File.Copy(file.FullName, destinationFile, false)
//                                    with
//                                    | _ ->  printToConasole.Post(sprintf "Error creating directory %s" dirDestination)
//                                    printToConasole.Post(sprintf "Copying %s..." destinationFile)
//                        with
//                        | ex ->  printToConasole.Post(sprintf "Error %s" ex.Message))
//
//    ignore( Console.ReadLine() )
//    0
