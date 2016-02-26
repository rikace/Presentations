namespace Easj360FSharp 

module EventEx = 
        open System.Threading
        open System.IO
        open System
        open System.Drawing
        open System.Windows.Forms
        open System.Windows.Controls

        // Initialize the watcher
        let w1 = new FileSystemWatcher("C:\\Temp", EnableRaisingEvents = true)

        // Test attributes of the file
        let isNotHidden(fse:FileSystemEventArgs) = 
          let hidden = FileAttributes.Hidden
          (File.GetAttributes(fse.FullPath) &&& hidden) <> hidden
    
        // Register the event handler
        let registerW1 =
              w1.Renamed.Add(fun fse ->
              // Report only visible files
              if isNotHidden(fse) then
                printfn "%s renamed to %s" fse.OldFullPath fse.FullPath)

        // --------------------------------------------------------------------------

        // Filtering events using Event.filter function 
        let w2 = new FileSystemWatcher("C:\\Temp", EnableRaisingEvents = true)

        // Filter renames of hidden files
        let renamedVisible = 
          w2.Renamed |> Event.filter isNotHidden
        // Print file name when event occurs
        let registerW2 =
            renamedVisible |> Event.add (fun fse -> 
            printfn "%s renamed to %s" fse.OldFullPath fse.FullPath)

// --------------------------------------------------------------------------

        // Function for formatting the information about event
        // (needed below in the listing 16.3)
        let formatFileEvent(fse:RenamedEventArgs) = 
          sprintf "%s renamed to %s" fse.OldFullPath fse.FullPath

        // Create a new file system watcher..
        let w3 = new FileSystemWatcher("C:\\Temp", EnableRaisingEvents = true)

        // Declarative event handling (F#)
        let registerW3 = 
            w3.Renamed 
              |> Event.filter isNotHidden
              |> Event.map formatFileEvent
              |> Event.add (printfn "%s")


module EventIOEx = 
        open System.Threading
        open System.IO
        open System

        type System.IO.Stream with
            member x.AsyncReadReport(count:int, ev:#Event<int>) =
                async { let! res = x.AsyncRead(count)
                        ev.Trigger(res.Length)      
                        return res  }

        type Test(fileName) =
            let ev = Event<int>()
            let fs = new FileStream(fileName, FileMode.Open, FileAccess.Read) :> Stream
            [<CLIEvent>]
            member x.Fire = ev.Publish
            member x.Read() = let action = async {
                                              let! ac = Async.StartChild(fs.AsyncReadReport(1024, ev))
                                              let! res = ac
                                              fs.Close()
                                              return res}
                              Async.RunSynchronously(action)

// EVENT FORM SAMPLE
module FormEventEx = 
        open System.Threading
        open System.IO
        open System
        open System.Drawing
        open System.Windows.Forms

        let fnt = new Font("Calibri", 24.0f)
        let lbl = new System.Windows.Forms.Label(Dock = DockStyle.Fill, TextAlign = ContentAlignment.MiddleCenter, Font = fnt)
        
        let form = new System.Windows.Forms.Form(ClientSize=Size(200, 100), Visible=true)
        
        do form.Controls.Add(lbl)
         
          // 'Infinite' asynchronous loop
        let rec loop(count) = async {       
            // Wait for the next click
            let! _ = Async.AwaitEvent(lbl.MouseDown)
            lbl.Text <- sprintf "Clicks: %d" count
            // OPTIONAL:  Wait 1 second to limit the clicking rate
            do! Async.Sleep(100)
            // Loop with incremented count
            return! loop(count + 1) }

          // Start the loop without blocking
        let start = Async.Start(loop(1))
  
  
          // NOTE: The following example is not discussed in the book!
          // Just for curiosity, this shows how the thing above
          // could be implemented using F# event combinators
        let regsiter = lbl.MouseDown 
                            |> Event.map (fun _ -> DateTime.Now)             // Create events carrying the current time
                            |> Event.scan (fun (_, dt:DateTime) ndt ->       // Remembers the last time click was accepted
                                if ((ndt - dt).TotalSeconds > 1.0) then      // When the time is more than a second...
                                  (1, ndt) else (0, dt)) (0, DateTime.Now)   // .. we return 1 and the new current time
                            |> Event.map fst |> Event.scan (+) 0             // Sum the yielded numbers 
                            |> Event.map (sprintf "Clicks: %d")              // Format the output as a string 
                            |> Event.add lbl.set_Text                     // Display the result...    


   type Watcher() =
        static member GoingAway(args : System.EventArgs) =
            System.Console.WriteLine("Going away now....")

    type Notify = delegate of string -> string

    type Child() =
        member this.Respond(msg : string) =
            System.Console.WriteLine("You want me to {0}? No!")
            "No!"

    type CurriedDelegate = delegate of int * int -> int
    type TupledDelegate = delegate of (int * int) -> int
    type DelegateTarget() =
        member this.CurriedAdd (x : int) (y : int) = x + y
        member this.TupledAdd (x : int, y : int) = x + y

    type ConcertEventArgs(city : string) =
        inherit System.EventArgs()
        member cea.City = city
        override cea.ToString() =
            System.String.Format("city:{0}", city)
    
    type RockBand(name : string) =
        let concertEvent = new DelegateEvent<System.EventHandler>()

        member rb.Name = name

        [<CLIEvent>]
        member rb.OnConcert = concertEvent.Publish
        member rb.HoldConcert(city : string) =
            concertEvent.Trigger([| rb; 
                new ConcertEventArgs(city) |])
            System.Console.WriteLine("Rockin' {0}!")

    type Fan(home : string, favBand : RockBand) as f =
        do
            favBand.OnConcert.AddHandler(
                System.EventHandler(f.FavoriteBandComingToTown))
        member f.FavoriteBandComingToTown 
                (_ : obj) 
                (args : System.EventArgs) =
            let cea = args :?> ConcertEventArgs
            if home = cea.City then
                System.Console.WriteLine("I'm SO going!")
            else
                System.Console.WriteLine("Darn")

//        let events_examples =
//            let ad = System.AppDomain.CurrentDomain
//            ad.ProcessExit.Add(Watcher.GoingAway)
//
//            let c = new Child()
//            let np = new Notify(c.Respond)
//            let response = np.Invoke("Clean your room!")
//            System.Console.WriteLine(response)
//    
//            let dt = new DelegateTarget()
//            let cd1 = new CurriedDelegate(dt.CurriedAdd)
//            //let cd2 = new CurriedDelegate(dt.TupledAdd) // will not compile
//            let td1 = new TupledDelegate(dt.TupledAdd)
//            //let td2 = new TupledDelegate(dt.CurriedAdd) // will not compile
//    
//            let rb = new RockBand("The Functional Ya-Yas")
//            rb.OnConcert 
//                |> Event.filter 
//                    (fun evArgs ->
//                        let cea = evArgs :?> ConcertEventArgs
//                        if cea.City = "Sacramento" then false
//                            // Nobody wants to tour in Sacramento
//                        else true)
//                |> Event.add
//                    (fun evArgs ->
//                        let cea = evArgs :?> ConcertEventArgs
//                        System.Console.WriteLine("{0} is rockin' {1}",
//                            rb.Name, cea.City))
//            let f1 = new Fan("Detroit", rb)
//            let f2 = new Fan("Cleveland", rb)
//            let f3 = new Fan("Detroit", rb)
//            rb.HoldConcert("Detroit")
//            ()
