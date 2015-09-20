namespace Easj360FSharp

open System
open System.IO

module SearchDataFile =

    type System.IO.Directory with
        static member AsyncGetFiles(path : string) =
            let dele = new Func<string * string, string[]>(Directory.GetFiles)
            Async.FromBeginEnd((path, "*.*"), dele.BeginInvoke, dele.EndInvoke)

        static member AsyncGetDirectories(path : string) =
            let dele = new Func<string * string, string[]>(Directory.GetDirectories)
            Async.FromBeginEnd((path, "*.*"), dele.BeginInvoke, dele.EndInvoke)

    type Search() =
        
        let seqLines file = seq { yield! File.ReadAllLines(file) }

        let parallel2 (job1, job2) =
            async { 
                    let! task1 = Async.StartChild job1
                    let! task2 = Async.StartChild job2
                    let! res1 = task1
                    let! res2 = task2
                    return (res1, res2) }

        let handleError e = printfn "%A" e
        let dataFile = ref (Seq.empty<string>)
        let dataDir = ref (Seq.empty<string>)
        let setDataFile fileSystems = dataFile := fileSystems
        let setDataDir fileSystems = dataDir := fileSystems

        let evaluete f = 
                    async {
                        return! Async.StartChild(f)
                        }
                            
        let rec getF p = 
            seq {
                let (files, dirs) = Async.RunSynchronously(parallel2(System.IO.Directory.AsyncGetFiles(p), System.IO.Directory.AsyncGetDirectories(p)))
                for f in files do yield f
                for d in dirs do yield! getF d

            }


//        let rec allFilesRec path = 
//                            seq {             
//                                    Async.StartWithContinuations(parallel2(System.IO.Directory.AsyncGetFiles(path), System.IO.Directory.AsyncGetDirectories(path)),
//                                                                (fun (f, d) -> setData((f|> Seq.append (d |> Seq.map allFilesRec |> Seq.concat)))),
//                                                                 handleError,
//                                                                 handleError)      
//                                    yield! !data       
//                                }                    

        let rec getFII p = 
            seq {
                Async.StartWithContinuations(parallel2(System.IO.Directory.AsyncGetFiles(p), System.IO.Directory.AsyncGetDirectories(p)),
                                                                (fun (f, d) -> do setDataFile f
                                                                               do setDataDir d),
                                                                 handleError,
                                                                 handleError)   
             //   let (files, dirs) = Async.RunSynchronously(parallel2(System.IO.Directory.AsyncGetFiles(p), System.IO.Directory.AsyncGetDirectories(p)))
                for f in !dataFile do yield f
                for d in !dataDir do yield! getFII d

            }

        member this.Start(path) =            
            seq{ for file in (getFII path) do yield file }


//             Seq.append  
//            (dir |> System.IO.Directory.GetFiles)
//            (dir |> System.IO.Directory.GetDirectories |> Seq.map allFilesRec |> Seq.concat)



    type 'a SearchResult(lineNo:int,file:string,content:'a) = 
        member this.LineNo = lineNo
        member this.File = file
        member this.Content = content
        override this.ToString() =
            sprintf "line %d in %s – %O" this.LineNo this.File this.Content
 
    let read (file:string) =
        seq { use reader = new StreamReader(file)
              while not reader.EndOfStream do yield reader.ReadLine() }

    let search_file parse check file =
        read file
        |> Seq.mapi (fun i l -> i + 1, parse l)
        |> Seq.filter check
        |> Seq.map (fun (i, l) -> SearchResult(i,  file, l))

    let search parse check =
        Seq.map (fun file -> search_file parse check file)
        >> Seq.concat
 
    let print_results<'a> : 'a SearchResult seq -> unit =    
        Seq.fold (fun c r -> printfn "%O" r; c + 1) 0
        >> printfn "%d results"

    //    Directory.GetFiles(@"c:\_src\", "*.cs", SearchOption.AllDirectories)
    //        |> search id (fun (i, l) -> l.Contains("public void Test"))
    //        |> print_results