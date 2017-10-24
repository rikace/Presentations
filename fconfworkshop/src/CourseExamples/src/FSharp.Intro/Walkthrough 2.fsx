open System
open System.IO

// --------------------------------------------------------
// Simple function demo
// --------------------------------------------------------

let sayHello printer = 
  printer "Hello"
  printer " "
  printer "world"

sayHello (fun msg ->
  printf "%s" msg)

sayHello (fun msg ->
  printf "%s" (msg.ToUpper()))


// --------------------------------------------------------
// Introducing functions via file system
// --------------------------------------------------------

// EXAMPLE: Enumerating files 

// Passing function as a parameter
let printFile file = 
  printfn "File: %s" file

let enumerateFiles operation root = 
  for file in Directory.GetFiles(root) do
    //printFile file  
    operation file  

enumerateFiles printFile "C:\\" 

// DEMO: Generalize 'enumerateFiles' to allow not just
// printing, but also other operations on files

let enumerateDirectories operation root = 
  for dir in Directory.GetDirectories(root) do
    //printfn "Directory: %s" dir
    operation dir

// TODO: Modify 'enumerateDirectories' similarly so that 
// we can call it with the following function as argument:

let printDirectory dir = 
  let name = Path.GetFileName(dir)
  let count = 
    try Seq.length(Directory.GetFiles(dir))
    with _ -> 0
  printfn "'%s' contains %d files" name count

enumerateDirectories printDirectory "C:\\Program Files" // 

// TODO: Create a new function called 'enumerateEntries' 
// that takes a directory and *two* functions. The first 
// one is called for all files and the other for all 
// directories. We want to be able to write:

let enumerateEntries f1 f2 dir =
   enumerateDirectories f1 dir
   enumerateFiles f2 dir

let rec enumerateEntriesRec fDir fFile dir =
    for file in Directory.GetFiles(dir) do fFile file
    for d in Directory.GetDirectories(dir) do 
        fDir d
        enumerateEntriesRec fDir fFile d

//   enumerateEntries "C:\\" printFile printDirectory

