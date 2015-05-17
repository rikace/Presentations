module PipelineImage

open System
open System.Diagnostics
open System.Threading
open System.Threading.Tasks
open System.Windows.Forms

open ImagePipeline.Image
open ImagePipeline.Extensions
open ImagePipeline.PipelineAgent
open ImagePipeline.Gui
open ImagePipeline.Main

type Mode =
    | Stopped = 0
    | Running = 1
    | Stopping = 2

/// Represents the program and implements GUI interaction
type Program() =
    let form = new ImagePipeline.Gui.MainForm()
    let sw = new Stopwatch()

    // Mutable state that is updated while showing images        
    let mutable imagesSoFar = 0
    let totalTime = [| 0; 0; 0; 0; 0; 0; 0; 0 |]

    // Stores radio button with associated pipeline mode 
    let buttons = 
      [ form.radioButtonMessagePassing, ImageMode.MessagePassing ]

    // A list of text box elements to show corresponding elements from 'totalTime'
    let infoboxes = 
      [ form.textBoxPhase1AvgTime; form.textBoxPhase2AvgTime; form.textBoxPhase3AvgTime;
        form.textBoxPhase4AvgTime; form.textBoxQueue1AvgWait; 
        form.textBoxQueue2AvgWait; form.textBoxQueue3AvgWait ]

    // --------------------------------------------------------------------------
    // GUI updates - functions that update various parts of the user interface
    // --------------------------------------------------------------------------

    /// Enables/disables buttons depending on whether application
    /// is running and selects the current pipeline mode
    let updateEnabledStatus mode imageMode =
        for btn, btnMode in buttons do 
            btn.Enabled <- (mode = Mode.Stopped)
            btn.Checked <- (btnMode = imageMode)
        form.buttonStart.Enabled <- (mode = Mode.Stopped)
        form.buttonStop.Enabled <- (mode = Mode.Running)
        form.quitButton.Enabled <- (mode = Mode.Stopped)

    /// Display processed image and update other information
    /// (such as average time, # of items in queues etc.)
    let setBitmap (imageInfo:ImageInfo) =
        // Display image & update count
        use priorImage = form.pictureBox1.Image
        form.pictureBox1.Image <- imageInfo.FilteredImage
        imageInfo.FilteredImage <- null
        imagesSoFar <- imagesSoFar + 1

        // Calculate duration of each phase
        for i in 0 .. 3 do 
            let time = imageInfo.PhaseEndTick.[i] - imageInfo.PhaseStartTick.[i]
            totalTime.[i] <- totalTime.[i] + time
 
        // Infer queue wait times by comparing phase(n+1) start with phase(n) finish timestamp
        for i in 0 .. 2 do
          totalTime.[4 + i] <- totalTime.[4 + i] + imageInfo.PhaseStartTick.[1 + i] 
                                                 - imageInfo.PhaseEndTick.[i]
        for i, box in infoboxes |> Seq.zip [0 .. 6] do 
          box.Text <- (totalTime.[i] / imagesSoFar).ToString()

        form.textBoxQueueCount1.Text <- imageInfo.QueueCount1.ToString()
        form.textBoxQueueCount2.Text <- imageInfo.QueueCount2.ToString()
        form.textBoxQueueCount3.Text <- imageInfo.QueueCount3.ToString()
        form.textBoxFileName.Text <- imageInfo.FileName
        form.textBoxImageCount.Text <- imageInfo.ImageCount.ToString()

        // Display FPS and verify image order
        let elapsedTime = sw.ElapsedMilliseconds
        form.textBoxFps.Text <- String.Format("{0: 0}", elapsedTime / int64 imageInfo.ImageCount)                 
        if imageInfo.SequenceNumber <> imagesSoFar - 1 then
            let msg = String.Format("Program error-- images are out of order. Expected {0} but received {1}",
                                    imagesSoFar - 1, imageInfo.SequenceNumber)
            MessageBox.Show(msg, "Application Error", MessageBoxButtons.OK, MessageBoxIcon.Asterisk) |> ignore
            Application.Exit()

    /// Call 'setBitmap' to update info on the GUI thread
    let updateFn (bm:ImageInfo) =
        form.Invoke(new Action(fun () -> setBitmap bm )) |> ignore

    /// Display error message when something goes wrong
    let errorFn (ex:Exception) =
        form.Invoke(new Action(fun () -> 
            MessageBox.Show(ex.Message, "Application Error", MessageBoxButtons.OK, MessageBoxIcon.Asterisk) 
              |> ignore ), ()) |> ignore

    // --------------------------------------------------------------------------
    // Reactive GUI - Waits for events & runs the pipeline processing
    // --------------------------------------------------------------------------

    /// Event that is triggered whenever a radiobutton (any) is checked
    let checkChanged = 
      [ for btn, _ in buttons -> btn.CheckedChanged :> IObservable<_> ]
      |> List.reduce Observable.merge

    
    /// Wait until the user selects 'Start' and handle radiobutton selection
    let rec waiting mode = async {
      let! evt = Async.AwaitObservable(checkChanged, form.buttonStart.Click)
      match evt with
      | Choice1Of2(_) ->
          // New pipeline mode was selected
          let mode = buttons |> List.pick (fun (btn, mode) ->
            if btn.Checked then Some(mode) else None)
          return! waiting mode
      | Choice2Of2(_) ->
          // Start pipeline processing
          return! processing mode }


    /// Starts the pipeline & waits until it is canceled and stops
    and processing imageMode = async {
      // Initialization & reset mutable values
      updateEnabledStatus Mode.Running imageMode
      let cts = new CancellationTokenSource()
      sw.Restart()
      imagesSoFar <- 0
      for i in 0 .. totalTime.Length - 1 do totalTime.[i] <- 0

      // Start the pipeline and create event that is triggered when it completes
      let pipelineCompleted = new Event<_>()
      let task = async { 
          ImagePipeline.Main.imagePipelineMainLoop updateFn cts.Token  errorFn  
          pipelineCompleted.Trigger() }
      Async.Start(task, cts.Token)
      
      // Wait until the user selects 'Stop' and trigger cancellation
      let! _ = form.buttonStop.Click |> Async.AwaitObservable
      updateEnabledStatus Mode.Stopping imageMode
      cts.Cancel() 

      // Wait until the pipeline actually completes and switch to waiting
      let! _ = pipelineCompleted.Publish |> Async.AwaitObservable
      updateEnabledStatus Mode.Stopped imageMode
      return! waiting imageMode }

    // Initialize GUI & start in the 'waiting' mode
    do 
      updateEnabledStatus Mode.Stopped ImageMode.MessagePassing
      form.quitButton.Click.Add(fun _ -> Application.Exit())
      waiting ImageMode.MessagePassing |> Async.StartImmediate

    member x.MainForm = form

// ------------------------------------------------------------------------------
// The main entry point for the application.
// ------------------------------------------------------------------------------

[<STAThread>]
do
    Application.EnableVisualStyles()
    Application.SetCompatibleTextRenderingDefault(false)
    let prog = new Program()
    Application.Run(prog.MainForm)
