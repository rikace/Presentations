module FileDuplicateFinder 
              
//#load "..\Threading\PSeq.fs"            

open System
open System.IO
open System.Security.Cryptography
open Microsoft.FSharp.Collections
open Microsoft.FSharp.Collections.Array.Parallel
open Microsoft.FSharp.Collections

//mp3
//mp4
//wma

let rec allFiles(dir) = seq {
    for file in System.IO.Directory.EnumerateFiles(dir) do yield file
    for sub in System.IO.Directory.EnumerateDirectories(dir) do yield! allFiles(sub) }

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

let sizePackage = 1024L

let partition condition values =     
    let pairs = seq {
        for i in values do
            if condition i then yield Some(i), None
            else yield None, Some(i) }
    pairs |> Seq.choose fst, pairs |> Seq.choose snd

let getAllFiles (dirs:string[]) =
    seq { for dir in dirs do yield! allFiles(dir) } 
   
let groupFiles (files:string seq) = //(f:(FileInfo -> 'a) when 'a : equality) =
    let (seqFileSizeBigger, seqFileSizeSmaller) = 
        files
        |> Seq.map (fun filePath -> new FileInfo(filePath)) 
        |> Seq.groupBy (fun (file:FileInfo) -> (file.Extension, file.Length))
        |> Seq.filter (fun (_, seqFiles) -> (Seq.length seqFiles) > 1)
        |> Seq.toArray
        |> Array.Parallel.partition (fun (key, _) -> (snd key) > sizePackage)
    ((seqFileSizeBigger |> Seq.map snd), (seqFileSizeSmaller |> Seq.map snd))   

let readBytesOffSet (file:FileInfo) = 
    use stream = file.OpenRead()
    let buff = Array.zeroCreate<byte> (int(sizePackage))
    let bytes = stream.Read(buff, 0, int(sizePackage))
    //let bytes = stream.cRead(int sizePackage)
    let hs = buff.[0..bytes-1] |> Convert.ToBase64String
    printfn "Computing Bytes file %s" file.Name
    (file, hs) 

let readMD5Hash (file:FileInfo) = 
    //let! hs = (Easj360FSharp.Hashing.hashFileAsync file.FullName)
    use stream = file.OpenRead()
    use hash = System.Security.Cryptography.MD5.Create()
    let bytes = hash.ComputeHash(stream)
    printfn "Computing Hash file %s" file.Name
    (file, (bytes |> Convert.ToBase64String))
            
let foundDuplicateFiles (f:(FileInfo -> (FileInfo * string))) (lst:seq<FileInfo seq>) =
            lst
            |> Seq.concat
            |> Seq.toArray
            |> Array.Parallel.map f
            |> Seq.groupBy (fun (file, bytes) -> bytes)
            |> Seq.filter (fun (_,s) -> Seq.length s > 1)
            |> Seq.map (fun (_, data) -> data |> Seq.map fst)
          
let (seqFileSizeBigger, seqFileSizeSmaller) = 
            allFiles @"\\GoFlex_Home\GoFlex Home Public\Immagini"//@"I:\Musica" 
            |> Seq.filter(fun f-> f.EndsWith(".mp3") || f.EndsWith(".mp4") || f.EndsWith(".wma"))
            |> groupFiles

//            |> Seq.filter(fun f -> f.Contains("_[0]_"))                        
//            |> Seq.iter (fun f -> printfn "Name %s" f
//                                  File.Delete f)

let duplicateFiles = 
            Async.Parallel [ async { return ((foundDuplicateFiles readBytesOffSet seqFileSizeBigger) |> foundDuplicateFiles readMD5Hash ) }; 
                             async { return (foundDuplicateFiles readMD5Hash seqFileSizeSmaller ) } ] 
            |> Async.RunSynchronously
            |> Seq.concat
            |> Seq.toArray
            |> Array.iter (fun d -> d |> Seq.skip 1
                                      |> Seq.iter (fun f -> 
                                                            printfn "Name %s" f.Name))

let fr = allFiles @"\\GoFlex_Home\GoFlex Home Public\Immagini"
        |> Seq.filter (fun f-> f.Contains("_[0]_"))
        |> Seq.map(fun f -> let index = f.IndexOf("_[0]_")
                            let fileNameToRemove = f.Remove(index, 5)
                            (f, fileNameToRemove))
        |> Seq.iter(fun (a, b) -> System.IO.File.Move(a, b))


let f1 = new FileInfo(@"I:\Musica\_VaryMp3\(11)-GreatStone.mp3")
let f2 = new FileInfo(@"I:\Musica\_VaryMp3\(11)-GreatStone_[0]_.mp3")
let b1 = File.ReadAllBytes(f1.FullName)
let b2 = File.ReadAllBytes(f2.FullName)


//let duplicateFiles = 
//            Async.Parallel [ async { return (foundDuplicateFiles seqFileSizeBigger readMD5Hash) }; 
//                             async { return (foundDuplicateFiles seqFileSizeSmaller readBytesOffSet) } ] 
//            |> Async.RunSynchronously
//            |> Seq.concat

//duplicateFiles |> Seq.iter (fun s -> Seq.iter (fun (file:FileInfo) -> printfn "file name %s" file.Name) s)
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

type StateProcess = CheckFirstPack  
                    | CheckLastPack
                    | CheckHash 
                    | Completed 

type FilesData = {  Size:int64                    
                    Files:seq<FileInfo>
                    InitialPackSize:int64
                    FinalPackSize:int64
                    State:StateProcess }

type ProcessDataGeneric (data:seq<FileInfo>) =
       
    [<LiteralAttribute>]
    let PACKSIZE = 1024L

    let createFilesDataCollection (size:int64, files:seq<FileInfo>) = 
        let initialPack, finalPack =
            if size = PACKSIZE || PACKSIZE > size then
                (size, size)
            elif size >= (PACKSIZE * 2L) then
                (PACKSIZE, PACKSIZE)
            else
                (PACKSIZE, (size - PACKSIZE))
        {   Size = size
            Files = files |> Seq.toList
            InitialPackSize = initialPack
            FinalPackSize = finalPack
            State = CheckFirstPack }
                                      
    let groupedDataByLen =
        seq { yield! (data  |> Seq.groupBy (fun file -> file.Length) 
                            |> Seq.map (fun (size, file) -> lazy (createFilesDataCollection(size, file)))) }

    let convert data subData = 
            data |> Seq.map (fun (_, subDataGrouped) -> { subData with Files = subDataGrouped  |> Seq.toList; State = match subData.State with 
                                                                                                                       | CheckFirstPack -> CheckLastPack
                                                                                                                       | CheckLastPack -> CheckHash
                                                                                                                       | CheckHash -> Completed 
                                                                                                                       | Completed -> failwith "Shouldn't reach this point" })  

    let createHashAsync (len:int64) offSet (fileInfo:FileInfo) = async {
            use hasher = new System.Security.Cryptography.SHA1CryptoServiceProvider()
            use stream = fileInfo.Open(FileMode.Open, FileAccess.Read, FileShare.ReadWrite)
            if len = -1L then 
                return ((hasher.ComputeHash(stream) |> Convert.ToBase64String), fileInfo)
            else
                stream.Seek(offSet, SeekOrigin.Begin) |> ignore
                let buffer = Array.zeroCreate<byte> (int(len))
                let! readCount = stream.AsyncRead buffer
                return ((hasher.ComputeHash(buffer) |> Convert.ToBase64String), fileInfo) }

    let compareAndGroupDataAsync (subData:FilesData) (f:(FileInfo -> Async<string * FileInfo>)) =
        let (duplicate, unique)  =  subData.Files
                                    |> Seq.map (fun file -> f(file))
                                    |> Async.Parallel
                                    |> Async.RunSynchronously
                                    |> Seq.groupBy fst
                                    |> Seq.map (fun (hash, info) -> (hash, (snd (info |> Seq.toList |> List.unzip))))
                                    |> Seq.toArray
                                    |> Array.partition (fun (_, groups) -> Seq.length groups > 1)
        ((convert duplicate subData), (convert unique subData))

    let processSearchFilesAsync (subData:seq<Lazy<FilesData>>) =     
        let rec processCloneFounder(filesToProcess:seq<Lazy<FilesData>>, acc:FilesData list)=  
            let (duplicateFilesToReprocess, uniqueFiles) =                                     
                                    filesToProcess                                             
                                    |> Seq.map (fun f -> let files = if f.IsValueCreated then f.Value else f.Force()
                                                         match files.State with                                     
                                                          | CheckFirstPack -> compareAndGroupDataAsync files (fun f -> createHashAsync files.InitialPackSize 0L f)
                                                          | CheckLastPack -> compareAndGroupDataAsync files (fun f -> createHashAsync files.FinalPackSize (files.Size - int64(files.FinalPackSize)) f)
                                                          | CheckHash -> compareAndGroupDataAsync files (fun f -> createHashAsync -1L 0L f)                                                           
                                                          | Completed -> failwith "Shouldn't reach this point" )
                                    |> Seq.toArray                                                             
                                    |> Array.unzip                                                              
                                                                                                                
            let (reprocess, fileCompletedAndIdentical) = duplicateFilesToReprocess                              
                                                        |> Seq.collect (fun files -> files)                    
                                                        |> Seq.toArray                                         
                                                        |> Array.Parallel.partition (fun f -> Seq.length f.Files > 1)
                                                        |> fst                                                       
                                                        |> Array.Parallel.partition (fun f -> f.State <> Completed)
            if reprocess.Length > 0 then                                                                           
               processCloneFounder(reprocess |> Seq.map Lazy.CreateFromValue, (acc @ (uniqueFiles |> Seq.collect  (fun files -> files) |> Seq.toList)))
            else
               ((fileCompletedAndIdentical |> Array.Parallel.map (fun f -> f.Files)), (acc |> Seq.collect (fun f -> f.Files) |> Seq.toList)) 
        processCloneFounder(subData, [])
                                        
    member x.Start(print) =             
        let (duplicate, unique) = processSearchFilesAsync(groupedDataByLen) 
        if print = true then duplicate |> Array.iter(fun d -> d |> Seq.iter(fun f ->  printfn "Name %s" f.Name))
        duplicate
        
   
let files = allFiles @"I:\Musica" 
            |> Seq.filter(fun f-> f.EndsWith(".mp3") || f.EndsWith(".mp4") || f.EndsWith(".wma"))
            |> Seq.map (fun file -> FileInfo(file))
let filesCompareTest = ProcessDataGeneric(files)
filesCompareTest.Start(true)

(*
        - load files
        - order by size (or something)
    - collection with 1 elements are ok
    - collection with more of 1 elements need refine
    - for each collection order by the first 1024 (or any size) bytes
    - collection with 1 elements are ok
    - collection with more of 1 elements need refine
    - for each collection order by last 1024 (or any size) bytes
    - collection with 1 elements are ok
    - collection with more of 1 elements need refine
    - for each collection order by hash
    - collection with 1 elements are ok
    - collection with more of 1 elements ARE EQUALS!!
*)


