namespace Easj360FSharp

open System
open System.IO

module GetFiles =

    let rec filesUnder basePath =
        seq {
            yield! Directory.GetFiles(basePath)
            for subDir in Directory.GetDirectories(basePath) do
                yield! filesUnder subDir 
        }

    
    let rec allFilesRec dir =
        Seq.append  
            (dir |> System.IO.Directory.GetFiles)
            (dir |> System.IO.Directory.GetDirectories |> Seq.map allFilesRec |> Seq.concat)
        
    let allFiles dir =
        seq { for file in System.IO.Directory.GetFiles(dir) do
                let creationTime = System.IO.File.GetCreationTime(file)
                yield (file,creationTime)
            }

    let rec getFiles dir filter =
        seq {   yield! System.IO.Directory.EnumerateFiles(dir, filter)
                for subDir in System.IO.Directory.EnumerateDirectories(dir) do
                    yield! getFiles subDir filter }
                    
