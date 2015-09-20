namespace Easj360FSharp 

 module GetFilesAsync = 
    open System
    open System.IO

    type System.IO.Directory with
        /// Retrieve all files under a path asynchronously
        static member AsyncGetFiles(path : string, searchPattern : string) =
            let dele = new Func<string * string, string[]>(Directory.GetFiles)
            Async.FromBeginEnd((path, searchPattern), dele.BeginInvoke, dele.EndInvoke)

        static member AsyncGetDirectories(path : string, searchPattern : string) =
            let dele = new Func<string * string, string[]>(Directory.GetDirectories)
            Async.FromBeginEnd((path, searchPattern), dele.BeginInvoke, dele.EndInvoke)

        
    type System.IO.File with
        /// Copy a file asynchronously
        static member AsyncCopy(source : string, dest : string) =
            let dele = new Func<string * string, unit>(File.Copy)
            Async.FromBeginEnd((source, dest), dele.BeginInvoke, dele.EndInvoke)


    let asyncBackup path searchPattern destPath =
        async {
            let! files = Directory.AsyncGetFiles(path, searchPattern)        
            for file in files do
                let filename = Path.GetFileName(file)
                do! File.AsyncCopy(file, Path.Combine(destPath, filename))
        }
