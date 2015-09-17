module ZipFiles

open ICSharpCode.SharpZipLib.Zip
open System.IO 

let readAllBytes(br:Stream) = 
    let len = 2048            
    let data = Array.zeroCreate len
    use ms = new MemoryStream()
    use bw = new BinaryWriter(ms)
    let mutable is_done = false
    while(not is_done) do
        let sz = br.Read(data, 0, len) 
        is_done <- (sz <= 0)
        if (sz > 0) then
            bw.Write(data, 0, sz)
    ms.ToArray()             
    
type FileEntry = {filename: string; contents: byte[]}
type ZipEntry = File of FileEntry
                | Dir of string

let fromZip (fileName: string): seq<ZipEntry> = 
    seq{
        use s = new ZipInputStream(File.OpenRead(fileName))
        let e = ref (s.GetNextEntry())
        while (!e <> null) do
            if (!e).IsFile then
                yield File {filename = (!e).Name; 
                            contents = readAllBytes s}
            else if (!e).IsDirectory then
                yield (Dir (!e).Name)
            e := s.GetNextEntry()}

let example () = 
    //dump names of all files in zip archive
    fromZip @"test.zip"
    |> Seq.choose (function (File f) -> Some f.filename
                            | _ -> None)
    |> Seq.iter (printfn "%A")