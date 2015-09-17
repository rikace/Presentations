namespace Easj360FSharp

open System
open System.Threading
open System.ComponentModel
open System.Windows.Forms
open System.Drawing.Imaging
open System.Drawing

module MailBoxCollector =

        // type that defines the messages types our updater can handle
        type Updates<'a> =
            | AddValue of 'a
            | GetValues of AsyncReplyChannel<list<'a>>
            | Stop

        // a generic collecter that recieves a number of post items and
        // once a configurable limit is reached fires the update even
        type Collector<'a>(?updatesCount) = 
            // the number of updates to cound to before firing the update even
            let updatesCount = match updatesCount with Some x -> x | None -> 100
    
            // Capture the synchronization context of the thread that creates this object. This
            // allows us to send messages back to the GUI thread painlessly.
            let context = AsyncOperationManager.SynchronizationContext
            let runInGuiContext f =
                context.Post(new SendOrPostCallback(fun _ -> f()), null)

            // This events are fired in the synchronization context of the GUI (i.e. the thread
            // that created this object)
            let event = new Event<list<'a>>() 

            let mailboxWorkflow (inbox: MailboxProcessor<_>) =
                // main loop to read from the message queue
                // the parameter "curr" holds the working data
                // the parameter "master" holds all values received
                let rec loop curr master = 
                    async { // read a message
                            let! msg = inbox.Receive()
                            match msg with
                            | AddValue x ->
                                let curr, master = x :: curr, x :: master
                                // if we have over 100 messages write
                                // message to the GUI
                                if List.length curr > updatesCount then
                                    do runInGuiContext(fun () -> event.Trigger(curr))
                                    return! loop [] master
                                return! loop curr master
                            | GetValues channel ->
                                // send all data received back
                                channel.Reply master
                                return! loop curr master
                            | Stop -> () } // stop by not calling "loop" 
                loop [] []

            // the mailbox that will be used to collect the data
            let mailbox = new MailboxProcessor<Updates<'a>>(mailboxWorkflow)
    
            // the API of the collector
    
            // add a value to the queue
            member w.AddValue (x) = mailbox.Post(AddValue(x))
            // get all the values the mailbox stores
            member w.GetValues() = mailbox.PostAndReply(fun x -> GetValues x)
            // publish the updates event
            [<CLIEvent>]
            member w.Updates = event.Publish
            // start the collector
            member w.Start() = mailbox.Start()
            // stop the collector
            member w.Stop() = mailbox.Post(Stop)

        // create a new instance of the collector
        let collector2 = new Collector<int*int*Color>()
        
        let width&height = 400
        // a form to display the updates
        let form2 = 
            // the bitmap that will hold the output data
            let bitmap = new Bitmap(width, height, PixelFormat.Format24bppRgb)
            let form = new Form(Width = width, Height = height, BackgroundImage = bitmap)
            // handle the collectors updates even and use it to post 
            collector2.Updates.Add(fun points -> 
                List.iter bitmap.SetPixel points
                form.Invalidate())
            // start the collector when the form loads
            form.Load.Add(fun _ -> collector2.Start())
            // when the form closes get all the values that were processed
            form.Closed.Add(fun _ -> 
                let vals = collector2.GetValues()
                MessageBox.Show(sprintf "Values processed: %i" (List.length vals))
                |> ignore
                collector2.Stop())
            form

        // start a worker thread running our fake simulation
        let startWorkerThread2() = 
            // function that loops infinitely generating random
            // "simulation" data
            let fakeSimulation() =
                let rand = new Random()
                let colors = [| Color.Red; Color.Green; Color.Blue |] 
                while true do
                    // post the random data to the collector
                    // then sleep to simulate work being done
                    collector2.AddValue(rand.Next(width), 
                        rand.Next(height), 
                        colors.[rand.Next(colors.Length)])
                    Thread.Sleep(rand.Next(100))
            // start the thread as a background thread, so it won't stop
            // the program exiting
            let thread = new Thread(fakeSimulation, IsBackground = true)
            thread.Start()

        // start 6 instances of our simulation
        let start2 = for _ in 0 .. 5 do startWorkerThread2()
                     form2.Show()


