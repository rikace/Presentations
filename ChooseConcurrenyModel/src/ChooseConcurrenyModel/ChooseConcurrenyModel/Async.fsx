// 10-15
// download and calculate length of info on homepage for.. 
open System.Net

let urlList = [ "Microsoft.com", "http://www.microsoft.com/" 
                "MSDN", "http://msdn.microsoft.com/" 
                "Google", "http://www.google.com"
              ]

let fetchAsync(name, url:string) =
    let uri = new System.Uri(url)

    async { 
        let webClient = new WebClient()
        let! html = webClient.AsyncDownloadString(uri)
        printfn "Read %d characters for %s" html.Length name
    }

let runAll() =
    urlList
    |> Seq.map fetchAsync
    |> Async.Parallel 
    |> Async.RunSynchronously
    |> ignore

runAll()



(*  Asynchronous programming is when you begin an operation on a background 
    thread to execute concurrently, and have it terminate at some later time; 
    either ignoring the new task to complete in the background, or polling it 
    at a later time to check on its return value    *)
#load "CommonModule.fsx"
open System
open System.IO
open System.Threading
open System.Net
open Common

module Version1 =  // No asynchrony yet, operations started explicitly

    let prog = 
        async {     printfn "Starting ..." 
                    return 1 }

    let result = prog |> Async.RunSynchronously


module Version2 =  // let!, non-blocking in F# Interactive, StartAsTask gets a .NET4.0 Task object

    let prog = async {  printfn "Starting ..." 
                        do! Async.Sleep(5000)
                        printfn "Finishing ..." 
                        return 1 }

    let result = prog |> Async.StartAsTask

    // Can evaluate in interactive while async is running
    1+1

    let  sendToTPL task = Async.StartAsTask <| async { return task }

module Version3 =  // asycns are first-class, parallelism by combining asyncs

    let prog(i) = async {   printfn "Starting %d ..." i
                            do! Async.Sleep((i%5) * 1000)
                            printfn "Finishing %d ..." i
                            return 1 }

    let result = 
        [1..10]
        |> Seq.map (fun i -> prog(i))
        |> Async.Parallel
        |> Async.StartAsTask

module Version4 =  // asyncs are very cheap, 100000 running concurrently, only a couple of threads used

    let prog(i) = async {   printfn "Starting %d ..." i
                            do! Async.Sleep((i%5) * 1000)
                            printfn "Finishing %d ..." i
                            return 1 }

    let result = 
        [1..100000]
        |> Seq.map (fun i -> prog(i))
        |> Async.Parallel
        |> Async.StartAsTask


module Vesrion5 =  

    let urls = [ "http://www.meetup.com/DC-fsharp"; 
                 "http://www.meetup.com/DC-fsharp"; 
                 "http://fsharp.org"]
    
    urls 
    |> List.map (fun url -> async {
            let client = new System.Net.WebClient()
            let! html = client.AsyncDownloadString(new System.Uri(url))
            printfn "Site len %d" html.Length })
    |> Async.Parallel
    |> Async.RunSynchronously
    


////////////// Async StartWithContinuation

let downloadComp (url : string) = async {
        let req = WebRequest.Create(url)
        let! rsp = req.AsyncGetResponse()
        use stream = rsp.GetResponseStream()
        use reader = new StreamReader(stream)
        return! reader.ReadToEndAsync() |> Async.AwaitTask }


let okCon (s: string) = printf "Length = %d\n" (s.Length) 
let exnCon _ = printf "Exception raised\n" 
let canCon _ = printf "Operation cancelled\n"

Async.StartWithContinuations
    (downloadComp "http://www.microsoft.com",
     okCon, exnCon, canCon);;

let cancelExample() =
    use ts = new CancellationTokenSource()
    Async.StartWithContinuations
        (downloadComp "http://www.dtu.dk",
         okCon, exnCon, canCon, ts.Token)
    ts.Cancel();;  

cancelExample();;

////////////// Async AwaitIAsyncResult
let streamWriter1 = File.CreateText("test1.txt")
let count = 10000000
let buffer = Array.init count (fun index -> byte (index % 256)) 

printfn "Writing to file test1.txt." 
let asyncResult = streamWriter1.BaseStream.BeginWrite(buffer, 0, count, null, null)

// Read a file, but use AwaitIAsyncResult to wait for the write operation 
// to be completed before reading. 
let readFile filename asyncResult count = 
    async { let! returnValue = Async.AwaitIAsyncResult(asyncResult)
            printfn "Reading from file test1.txt." 
            // Close the file.
            streamWriter1.Close()
            // Now open the same file for reading. 
            let streamReader1 = File.OpenText(filename)
            let! newBuffer = streamReader1.BaseStream.AsyncRead(count)
            return newBuffer }

let bufferResult = readFile "test1.txt" asyncResult count
                   |> Async.RunSynchronously

////////// Composition

let task1 = async { do! Async.Sleep 2000
                    return 1 }
let task2 = async { printfn "Waiting 10 seconds..."
                    for i = 1 to 10 do
                        printfn "%d..." i
                        do! Async.Sleep(1000)
                    printfn "Finished!" }

let ts' = new System.Threading.CancellationTokenSource()

Async.StartWithContinuations(
            Async.Parallel2(task1, task2),
            (fun res ->printfn "Operation completed ( %d )" (fst res)),
            (fun ex -> printfn "Operation failed."),
            (fun can -> printfn "Operation canceled."), ts'.Token)

ts'.Cancel()

//////// Fast Copy Async
open System.IO

let copy (source:System.IO.Stream) (destination:System.IO.Stream) = async {
        source.Position <- 0L
        destination.Position <- 0L

        let buffer = Array.init 2 (fun _ -> Array.zeroCreate<byte> 4096)
        let index = ref 1
        let! read = source.AsyncRead(buffer.[!index])
        if read > 0 then 
            let rec copyAsync (s:System.IO.Stream) (d:System.IO.Stream) (buff:byte[][]) read = async {
                let! (_, read') = Async.Parallel2(d.AsyncWrite(buff.[!index], 0, read), s.AsyncRead(buff.[(!index ^^^ 1)]))
                if read' > 0 then 
                    index := !index ^^^ 1
                    return! copyAsync s d buff read' }
            return! copyAsync source destination buffer read
        return (source, destination)}

let fileSource = System.IO.Path.Combine(System.IO.Path.GetDirectoryName(__SOURCE_DIRECTORY__ ),"Data\\Images\Bugghina1.jpg")

let fileDest = System.IO.Path.ChangeExtension(System.IO.Path.GetTempFileName(), ".jpg")

let copyAsync() = 
    let streamSource = new FileStream(fileSource, FileMode.Open, FileAccess.Read,FileShare.Read,0x1000, true)
    let streamDest = new FileStream(fileDest, FileMode.Create,FileAccess.Write, FileShare.Write,0x1000,true)
    
    copy streamSource streamDest 
        |> Async.RunSynchronously 
        ||> (fun s d -> s.Dispose()
                        d.Dispose())
copyAsync()

let buffer1 = File.ReadAllBytes(fileSource)
let buffer2 = File.ReadAllBytes(fileDest)

Seq.compareWith(fun x y -> compare x y) buffer1 buffer2 = 0

System.Diagnostics.Process.Start(fileSource)
System.Diagnostics.Process.Start(fileDest)

//////////////////////////////////////////

let wc = new WebClient()

wc.DownloadProgressChanged.Add(
  fun args -> printfn "Percent complete %d" args.ProgressPercentage)

wc.DownloadStringCompleted
  |> Event.filter(fun args -> args.Error = null && not args.Cancelled)  
  |> Event.map(fun args -> args.Result)
  |> Event.add(printfn "%s")

let fileSource' = System.IO.Path.Combine(System.IO.Path.GetDirectoryName(__SOURCE_DIRECTORY__ ),"Data\\DocTest.txt")
wc.AsyncDownloadString(new Uri("file://"+fileSource')) |> Async.RunSynchronously |> ignore

let bufferData = Array.zeroCreate<byte> 100000000

let async1 (label:System.Windows.Forms.Label) filename =
     Async.StartWithContinuations(
         async {    label.Text <- "Operation started." 
                    use outputFile = System.IO.File.Create(filename)
                    do! outputFile.AsyncWrite(bufferData)   },
         (fun _ -> label.Text <- "Operation completed."),
         (fun _ -> label.Text <- "Operation failed."),
         (fun _ -> label.Text <- "Operation canceled."))

// ~~~~~~~~~~~~  Fetch FILE

let getFile filePath = async {
    use stream = File.OpenRead filePath
    use reader = new StreamReader(stream)
    return! reader.ReadToEndAsync() |> Async.AwaitTask }

let files = Directory.GetFiles("..\Data", "*.csv")

files
|> Array.map getFile
|> Async.Parallel
|> Async.RunSynchronously
         
// ~~~~~~~~~~~~  Fetch HTML

let getHtml (url : string) = async {
        let req = WebRequest.Create(url)
        let! rsp = req.AsyncGetResponse()
        use stream = rsp.GetResponseStream()
        use reader = new StreamReader(stream)
        return! reader.ReadToEndAsync() |> Async.AwaitTask }
    
let html =getHtml "http://en.wikipedia.org/wiki/F_Sharp_programming_language"
          |> Async.RunSynchronously


let webPages : string[] =
    [ "http://www.bing.com"; "http://www.google.com"; "http://www.yahoo.com" ]
    |> List.map getHtml
    |> Async.Parallel
    |> Async.RunSynchronously



