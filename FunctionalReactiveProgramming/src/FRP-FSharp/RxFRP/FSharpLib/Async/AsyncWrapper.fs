namespace Easj360FSharp

module AsyncWrapper =

        let  sendToChildThread task = 
            async { let! job = Async.StartChild task
                    let! res = job
                    return res
                    }

        let  sendToThread task = 
              Async.Start <| async { return task }    
    
        let  sendToTPL task = 
              Async.StartAsTask <| async { return task }



(*
        open System.IO

        // Easy Wrapper for thread pool work
        let  sendToTPL task = 
              Async.StartAsTask <| async { return task }

        // Just add:      
        //       |>  sendToTPL  
        // to the end of a Let expression that can be run in the background 
        // while other work is performed

        // 
        // Simple example of loading three large text data files
        // 
        let AlphaFileName = "c:\somewhere"
        let BetaFileName  = "c:\somewhere"
        let DeltaFileName = "c:\somewhere"


        let placeFileIntoArray = File.ReadAllLines
        
        // Load/Parse files
      
        let FileAlphaLines = AlphaFileName    
                                     |> placeFileIntoArray 
                                     |> sendToTPL
        let FileBetaLines  = BetaFileName          
                                     |> placeFileIntoArray 
                                     |> sendToTPL  
        let FileDeltaLines = DeltaFileName        
                                     |> placeFileIntoArray 
                                     |> sendToTPL
         
        // Do other work here
        // ...

        // Creates header -> column index Map using given delimiter
        let CreateHeadersMap fileslines delimiter =
             // not important for this example
             0 // added so example compiles

        // Later use .Result to get the work results when you need them
        
        let FileAlphaHeaderMap  = CreateHeadersMap FileAlphaLines.Result [|'\t'|]
        let FileBetaHeadersMap  = CreateHeadersMap FileBetaLines.Result  [|','|]
        let FileDeltaHeadersMap = CreateHeadersMap FileDeltaLines.Result [|'\t'|] 
*)