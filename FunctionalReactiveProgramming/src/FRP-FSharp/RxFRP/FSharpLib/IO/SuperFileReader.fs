namespace Easj360FSharp 
          
open System

module SuperFileReader = 
 
    type SuperFileReader() =
        let progressChanged = new Event<int>()
 
        member this.ProgressChanged = progressChanged.Publish
 
        member this.OpenFile (filename : string, charsPerBlock) =
            use sr = new System.IO.StreamReader(filename)
            let streamLength = int64 sr.BaseStream.Length
            let sb = new System.Text.StringBuilder(int streamLength)
            let charBuffer = Array.zeroCreate<char> charsPerBlock
 
            let mutable oldProgress = 0
            let mutable totalCharsRead = 0
            progressChanged.Trigger(0)
            while not sr.EndOfStream do
                (* sr.ReadBlock returns number of characters read from stream *)
                let charsRead = sr.ReadBlock(charBuffer, 0, charBuffer.Length)
                totalCharsRead <- totalCharsRead + charsRead
 
                (* appending chars read from buffer *)
                sb.Append(charBuffer, 0, charsRead) |> ignore
 
                let newProgress = int(decimal totalCharsRead / decimal streamLength * 100m)
                if newProgress > oldProgress then
                    progressChanged.Trigger(newProgress) // passes newProgress as state to callbacks
                    oldProgress <- newProgress
 
            sb.ToString()
 
    let fileReader = new SuperFileReader()
    fileReader.ProgressChanged.Add(fun percent ->
        printfn "%i percent done..." percent)
 
    let x = fileReader.OpenFile(@"E:\SourceCode_Book\FSharp_Programming_BookOreilly\Ch02.fsx", 50)
    printfn "%s[...]" x.[0 .. if x.Length <= 100 then x.Length - 1 else 100]



