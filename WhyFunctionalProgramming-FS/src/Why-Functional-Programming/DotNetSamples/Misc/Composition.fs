namespace Misc
open System
open System.IO

module Compostion =

    let dir = @"c:\Temp"

    // 1
    let sizeOfFolder folder =

        // Get all files under the path
        let filesInFolder : string [] =
            Directory.GetFiles(folder, "*.*", SearchOption.AllDirectories)

        // Map those files to their corresponding FileInfo object
        let fileInfos : FileInfo [] = Array.map (fun (file : string) -> new FileInfo(file))
                                                filesInFolder

        // Map those fileInfo objects to the file's size
        let fileSizes : int64 [] = Array.map (fun (info : FileInfo) -> info.Length)
                                             fileInfos

        // Total the file sizes
        let totalSize = Array.sum fileSizes

        // Return the total size of the files
        totalSize

    printfn "Folder Size %d" (sizeOfFolder dir)


    // 2
    let sizeOfFolderPiped2 folder =
        let getFiles folder =
            Directory.GetFiles(folder, "*.*", SearchOption.AllDirectories)

        // the pipe operator is synthetic sugar  
        // give a function "f" and any parameter "g" apply "f(g)"
        // |> signature ('a -> ('a -> 'b) -> 'b) 
        
        folder
        |> getFiles
        |> Array.map (fun file -> new FileInfo(file))
        |> Array.map (fun info -> info.Length)
        |> Array.sum

    printfn "Folder Piped Size %d" (sizeOfFolder dir)

    
    // 3
    let sizeOfFolderComposed = // Note! no parameters!! 

        let log x = printfn "file name %A" x; x
        let getFiles folder =
            Directory.GetFiles(folder, "*.*", SearchOption.AllDirectories)

        //  >> signature (('a -> 'b) -> ('b -> 'c) -> 'a -> 'c)

        let filterMapAndSum = 
                    Array.filter (fun f -> Path.GetExtension(f) = ".docx") 
                   // log
                    >> Array.map (fun file -> new FileInfo(file))
                    >> Array.map (fun info -> info.Length)
                    >> Array.sum
        let sizeOfFolderComposed' = getFiles >> filterMapAndSum

        // The result of this expression is a function that takes
        // one parameter, which will be passed to getFiles and piped
        // through the following functions.
        getFiles
        >> Array.filter (fun f -> Path.GetExtension(f) = ".ps1") 
        >> log
        >> Array.map (fun file -> new FileInfo(file))
        >> Array.map (fun info -> info.Length)
        >> Array.sum

    printfn "Folder Composed Size %d" (sizeOfFolderComposed dir)



