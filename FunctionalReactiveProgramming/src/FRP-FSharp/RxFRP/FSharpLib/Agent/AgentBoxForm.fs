namespace Easj360FSharp

open System
open System.Threading
open System.ComponentModel
open System.Windows.Forms
open System.Drawing.Imaging
open System.Drawing

module AgentBoxForm = 

        // the width & height for the simulation
        let width, height = 500, 600

        // the bitmap that will hold the output data
        let bitmap = new Bitmap(width, height, PixelFormat.Format24bppRgb)

        // a form to display the bitmap
        let form = new Form(Width = width, Height = height,
                            BackgroundImage = bitmap)

        // the function which recieves that points to be plotted
        // and marshals to the GUI thread to plot them
        let printPoints points =
            form.Invoke(new Action(fun () -> 
                List.iter bitmap.SetPixel points
                form.Invalidate())) 
            |> ignore

        // the mailbox that will be used to collect the data
        let mailbox = 
            MailboxProcessor.Start(fun mb ->
                // main loop to read from the message queue
                // the parameter "points" holds the working data
                let rec loop points =
                    async { // read a message
                            let! msg = mb.Receive()
                            // if we have over 100 messages write
                            // message to the GUI
                            if List.length points > 100 then
                                printPoints points
                                return! loop []
                            // otherwise append message and loop
                            return! loop (msg :: points) }
                loop [])

        // start a worker thread running our fake simulation
        let startWorkerThread() = 
            // function that loops infinitely generating random
            // "simulation" data
            let fakeSimulation() =
                let rand = new Random()
                let colors = [| Color.Red; Color.Green; Color.Blue |] 
                while true do
                    // post the random data to the mailbox
                    // then sleep to simulate work being done
                    mailbox.Post(rand.Next(width), 
                        rand.Next(height), 
                        colors.[rand.Next(colors.Length)])
                    Thread.Sleep(rand.Next(100))
            // start the thread as a background thread, so it won't stop
            // the program exiting
            let thread = new Thread(fakeSimulation, IsBackground = true)
            thread.Start()

        // start 6 instances of our simulation
        let start = for _ in 0 .. 5 do startWorkerThread()
                    // run the form
                    form.Show()

