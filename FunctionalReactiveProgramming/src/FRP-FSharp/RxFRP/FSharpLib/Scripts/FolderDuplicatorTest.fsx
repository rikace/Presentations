#light

open System
open System.Threading
open System.IO

[<LiteralAttribute>]
let SOURCE = @"e:\Source"

[<LiteralAttribute>]
let TARGET = @"e:\Target"

[<LiteralAttribute>]
let MIRROR = @"e:\Mirror"

let rec allFiles(dir) = seq {
    for file in System.IO.Directory.EnumerateFiles(dir) do yield file
    for sub in System.IO.Directory.EnumerateDirectories(dir) do yield! allFiles(sub) }

let cleanTarget = Directory.Delete(TARGET, true)
                  //   allFiles(TARGET) |> Seq.iter(fun f -> File.Delete(f))

//let compareDirs = 
//    let filesTarget = 
//        allFiles(TARGET)
//        |> Seq.map (fun f -> new FileInfo(f))
//        |> Seq.sortBy(fun f -> (f.FullName, f.Length))
//    let filesMirror = 
//        allFiles(MIRROR)
//        |> Seq.map (fun f -> new FileInfo(f))
//        |> Seq.sortBy(fun f -> (f.FullName, f.Length))
//    if (Seq.length filesMirror) = (Seq.length filesTarget) then 
//        let comp = Seq.compareWith(fun (fm:FileInfo) (ft:FileInfo) ->   if ft.Name = fm.Name && ft.Length = fm.Length then 0
//                                                                        else -1)
//        (comp filesMirror filesTarget) = 0
//    else false

let copy s d =  
    if not(Directory.Exists(d)) then
        Directory.CreateDirectory(d) |> ignore
    for f in allFiles(s) do
        let newName = Path.Combine(d, Path.GetFileName(f))
        if not(Directory.Exists(Path.GetDirectoryName(newName))) then
            Directory.CreateDirectory(Path.GetDirectoryName(newName)) |> ignore
        File.Copy(f, newName, true)


let rename s =
     allFiles(TARGET)
     |> Seq.iter(fun f -> File.Move(f, (sprintf "%s\\%s%s%s" (Path.GetDirectoryName(f)) (Path.GetFileNameWithoutExtension(f)) (DateTime.Now.Ticks.ToString()) (Path.GetExtension(f)))))

let delete s =
    allFiles(TARGET)
    |> Seq.iter (fun f -> File.Delete(f))
    

copy SOURCE TARGET
rename TARGET