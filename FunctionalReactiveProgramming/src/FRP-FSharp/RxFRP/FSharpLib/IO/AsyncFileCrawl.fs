namespace Easj360FSharp

module AsyncFileCrawl =

    open System
    open System.IO
    open System.Threading.Tasks

    let private readFileAsync (file:string) (f:byte[] -> 'a) =
      async {
        // Open stream
        use stream = File.OpenRead(file)

        // Async read, so we don't block on the thread pool
        let! content = stream.AsyncRead(int stream.Length)

        // Apply operation on data and return
        return f content
      } |> Async.StartAsTask

    let rec private readDir (dir:string) (f:byte[] -> 'a) =
      seq {
        // Fire of one async read per file
        for file in Directory.GetFiles(dir) do
          yield readFileAsync file f

        // Spider down through the directory structure
        for dir in Directory.GetDirectories(dir) do
          yield! readDir dir f
      }

    let readFilesAsync<'a> (dir:string) (f:byte[] -> 'a) =
      let tasks = 
        readDir dir f 
        |> Seq.cast<Task> 
        |> Array.ofSeq

      // Wait for all tasks to complete
      Task.WaitAll(tasks)

      // Return result as a seq
      seq { 
        for task in tasks do 
          yield (task :?> Task<'a>).Result 
      }

    let result = 
      readFilesAsync<int> @"C:\Users\Fredrik\Projects\IronJS" (fun b -> b.Length)

