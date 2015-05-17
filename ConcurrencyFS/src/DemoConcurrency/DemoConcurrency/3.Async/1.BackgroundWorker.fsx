open System
open System.Threading
open System.ComponentModel
open System.Windows.Forms

/// Background workers (1)
let worker1 = new BackgroundWorker()

worker1.DoWork.Add(fun args ->    
    for i in 1 .. 50 do
        Thread.Sleep(1000)
        printfn "%A" i )

worker1.DoWork.Add(fun args ->  for i in 1 .. 10 do
                                    Thread.Sleep(500)
                                    printfn "B: %A" i)

worker1.RunWorkerCompleted.Add(fun args -> printfn "Completed...")
worker1.RunWorkerAsync()


/// Background workers (2)
let workerCancel = new BackgroundWorker(WorkerSupportsCancellation = true)
workerCancel.DoWork.Add(fun args ->
            printfn "apan %A" args
            for i in 1 .. 50 do
                if (workerCancel.CancellationPending = false) then
                    Thread.Sleep(1000)
                    printfn "%A" i  )

workerCancel.RunWorkerCompleted.Add(fun args -> printfn "Completed...")

workerCancel.RunWorkerAsync()
workerCancel.CancelAsync()



let worker = new BackgroundWorker()
let numIterations = 1000

worker.DoWork.Add(fun args ->

    let rec computeFibonacci resPrevPrev resPrev i =        
        let res = resPrevPrev + resPrev
        
        if i = numIterations then
            args.Result <- box res
        else
            computeFibonacci resPrev res (i + 1)

    computeFibonacci 1 1 2)

worker.RunWorkerCompleted.Add(fun args -> printfn "Result = %A" args.Result) 

worker.RunWorkerAsync()


/// An IterativeBackgroundWorker follows the BackgroundWorker design pattern
type IterativeBackgroundWorker<'T>(oneStep : ('T -> 'T),
                                   initialState : 'T,
                                   numIterations : int) =
    let worker =
        new BackgroundWorker(WorkerReportsProgress = true,
                             WorkerSupportsCancellation = true)

    let completed = new Event<_>()
    let error = new Event<_>()
    let cancelled = new Event<_>()
    let progress = new Event<_>()

    do worker.DoWork.Add(fun args ->
        let rec iterate state i =
            if worker.CancellationPending then
               args.Cancel <- true
            elif i < numIterations then
                let state' = oneStep state
                let percent = int ((float (i + 1) / float numIterations) * 100.0)
                do worker.ReportProgress(percent, box state);
                iterate state' (i + 1)
            else
                args.Result <- box state

        iterate initialState 0)

    do worker.RunWorkerCompleted.Add(fun args ->
        if args.Cancelled then cancelled.Trigger()
        elif args.Error <> null then error.Trigger args.Error
        else completed.Trigger (args.Result :?> 'T))

    do worker.ProgressChanged.Add(fun args ->
        progress.Trigger (args.ProgressPercentage,(args.UserState :?> 'T)))

    member x.WorkerCompleted = completed.Publish
    member x.WorkerCancelled = cancelled.Publish
    member x.WorkerError = error.Publish
    member x.ProgressChanged = progress.Publish
    member x.RunWorkerAsync() = worker.RunWorkerAsync()
    member x.CancelAsync() = worker.CancelAsync()

let fibOneStep (fibPrevPrev : bigint, fibPrev) = (fibPrev, fibPrevPrev + fibPrev);;
let worker = new IterativeBackgroundWorker<_>(fibOneStep, (1I, 1I), 100);;

worker.WorkerCompleted.Add(fun result -> printfn "Result = %A" result)

worker.ProgressChanged.Add(fun (percentage, state) ->
    printfn "%d%% complete, state = %A" percentage state)

worker.RunWorkerAsync()


// Winform Test
open System.Drawing
open System.Windows.Forms

let form = new Form(Visible = true, TopMost = true)

let panel = new FlowLayoutPanel(Visible = true,
                                Height = 20,
                                Dock = DockStyle.Bottom,
                                BorderStyle = BorderStyle.FixedSingle)

let progress = new ProgressBar(Visible = false,
                               Anchor = (AnchorStyles.Bottom ||| AnchorStyles.Top),
                               Value = 0)

let text = new Label(Text = "Paused",
                     Anchor = AnchorStyles.Left,
                     Height = 20,
                     TextAlign = ContentAlignment.MiddleLeft)

panel.Controls.Add(progress)
panel.Controls.Add(text)
form.Controls.Add(panel)

let fibOneStep (fibPrevPrev : bigint, fibPrev) = (fibPrev, fibPrevPrev + fibPrev)
let rec repeatMultipleTimes n f s = 
    if n <= 0 then s else repeatMultipleTimes (n - 1) f (f s)


let rec burnSomeCycles n f s = 
    if n <= 0 then f s else ignore (f s); burnSomeCycles (n - 1) f s

let step = (repeatMultipleTimes 500 (burnSomeCycles 1000 fibOneStep))
let worker = new IterativeBackgroundWorker<_>(step, (1I, 1I), 80)

worker.ProgressChanged.Add(fun (progressPercentage, state)->
    progress.Value <- progressPercentage)

worker.WorkerCompleted.Add(fun (_, result) ->
    progress.Visible <- false;
    text.Text <- "Paused";
    MessageBox.Show(sprintf "Result = %A" result) |> ignore)

worker.WorkerCancelled.Add(fun () ->
    progress.Visible <- false;
    text.Text <- "Paused";
    MessageBox.Show(sprintf "Cancelled OK!") |> ignore)

worker.WorkerError.Add(fun exn ->
    text.Text <- "Paused";
    MessageBox.Show(sprintf "Error: %A" exn) |> ignore)

form.Menu <- new MainMenu()
let workerMenu = form.Menu.MenuItems.Add("&Worker")

workerMenu.MenuItems.Add(new MenuItem("Run", onClick = (fun _ args ->
    text.Text <- "Running";
    progress.Visible <- true;
    worker.RunWorkerAsync())))

workerMenu.MenuItems.Add(new MenuItem("Cancel", onClick = (fun _ args ->
    text.Text <- "Cancelling";
    worker.CancelAsync())))

form.Closed.Add(fun _ -> worker.CancelAsync())