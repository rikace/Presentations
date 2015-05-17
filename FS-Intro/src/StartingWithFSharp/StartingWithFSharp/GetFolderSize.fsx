open System
open System.IO


let folderPath = @"c:\Temp"

let sizeOfFolder folder =

    // Get all files under the path
    let filesInFolder : string []  = 
        Directory.GetFiles(
            folder, "*.*", 
            SearchOption.AllDirectories)

    // Map those files to their corresponding FileInfo object
    let fileInfos : FileInfo [] = 
        Array.map 
            (fun file -> new FileInfo(file)) 
            filesInFolder

    // Map those fileInfo objects to the file's size
    let fileSizes : int64 [] = 
        Array.map 
            (fun (info : FileInfo) -> info.Length) 
            fileInfos

    // Total the file sizes
    let totalSize = Array.sum fileSizes

    // Return the total size of the files
    totalSize

sizeOfFolder folderPath

// ----------------------------------------------------------------------------

let sizeOfFolderPiped folder =

    let getFiles folder = 
        Directory.GetFiles(folder, "*.*", SearchOption.AllDirectories)
    
    let totalSize =
        folder
        |> getFiles
        |> Array.map (fun file -> new FileInfo(file)) 
        |> Array.map (fun info -> info.Length)
        |> Array.sum

    totalSize

sizeOfFolderPiped folderPath
// ----------------------------------------------------------------------------

let sizeOfFolderComposed (*No Parameters!*) =

    let getFiles folder =
        Directory.GetFiles(folder, "*.*", SearchOption.AllDirectories)

    // The result of this expression is a function that takes
    // one parameter, which will be passed to getFiles and piped
    // through the following functions.
    getFiles
    >> Array.map (fun file -> new FileInfo(file))
    >> Array.map (fun info -> info.Length)
    >> Array.sum

sizeOfFolderComposed folderPath




// ----------------------------------------------------------------------------

// Combining active patterns 

let (|KBInSize|MBInSize|GBInSize|) filePath =
    let file = File.Open(filePath, FileMode.Open)
    if   file.Length < 1024L * 1024L then
        KBInSize
    elif file.Length < 1024L * 1024L * 1024L then
        MBInSize
    else
        GBInSize

let (|EndsWithExtension|_|) ext file = if Path.GetExtension(file) = ext then Some()
                                       else None

let (|IsImageFile|_|) filePath =
    match filePath with
    | EndsWithExtension ".jpg"
    | EndsWithExtension ".bmp"
    | EndsWithExtension ".gif"
        -> Some()
    | _ -> None

let ImageTooBigForEmail filePath =
    match filePath with
    | IsImageFile & (MBInSize | GBInSize) -> true
    | _ -> false


let sizeOfFolderComposed' filter =

    let getFiles folder =
        Directory.GetFiles(folder, "*.*", SearchOption.AllDirectories)

    // The result of this expression is a function that takes
    // one parameter, which will be passed to getFiles and piped
    // through the following functions.
    getFiles
    >> Array.filter filter
    >> Array.map (fun file -> new FileInfo(file))
    >> Array.map (fun info -> info.Length)
    >> Array.sum

sizeOfFolderComposed' ImageTooBigForEmail folderPath
