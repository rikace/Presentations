open System;
open System.IO;

// get the total file size in the delcared directory

let dir = @"c:\Temp"

let files = Directory.GetFiles(dir, "*.*", SearchOption.AllDirectories)

let getFileSizeImp files = 
    let mutable total = 0L
    for file in files do
        if Path.GetExtension(file) = ".ps1" then
            let file' = new FileInfo(file)
            total <- total + file'.Length
    total 

getFileSizeImp files

let log x = printfn "file name %A" x; x

let getFileSizeFP =
   Array.filter (fun f -> Path.GetExtension(f) = ".ps1") 
   >> Array.map (fun f -> new FileInfo(f)) 
   >> log
   >> Array.map (fun (f:FileInfo) -> f.Length) 
   >> Array.sum 

getFileSizeFP files