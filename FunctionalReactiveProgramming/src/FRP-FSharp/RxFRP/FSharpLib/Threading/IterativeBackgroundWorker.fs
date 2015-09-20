namespace Easj360FSharp 

open System.ComponentModel
open System.Threading
open System.ComponentModel
open System.Windows.Forms

module IterativeBackgroundWorker =
    /// An IterativeBackgroundWorker follows the BackgroundWorker design pattern
    /// but instead of running an arbitrary computation it iterates a function
    /// a fixed number of times and reports intermediate and final results.
    /// The worker is paramaterized by its internal state type.
    ///
    /// Percentage progress is based on the iteration number. Cancellation checks
    /// are made at each iteration. Implemented via an internal BackgroundWorker.
    type IterativeBackgroundWorker<'T>(oneStep:('T -> 'T),
                                       initialState:'T,
                                       numIterations:int) =

        let worker =
            new BackgroundWorker(WorkerReportsProgress=true,
                                 WorkerSupportsCancellation=true)


        // Create the events that we will later trigger
        let completed = new Event<_>()
        let error     = new Event<_>()
        let cancelled = new Event<_>()
        let progress  = new Event<_>()

        do worker.DoWork.Add(fun args ->
            // This recursive function represents the computation loop.
            // It runs at "maximum speed", i.e. is an active rather than
            // a reactive process, and can only be controlled by a
            // cancellation signal.
            let rec iterate state i =
                // At the end of the computation terminate the recursive loop
                if worker.CancellationPending then
                   args.Cancel <- true
                elif i < numIterations then
                    // Compute the next result
                    let state' = oneStep state

                    // Report the percentage computation and the internal state
                    let percent = int ((float (i+1)/float numIterations) * 100.0)
                    do worker.ReportProgress(percent, box state);

                    // Compute the next result
                    iterate state' (i+1)
                else
                    args.Result <- box state

            iterate initialState 0)

        do worker.RunWorkerCompleted.Add(fun args ->
            if args.Cancelled       then cancelled.Trigger()
            elif args.Error <> null then error.Trigger args.Error
            else completed.Trigger (args.Result :?> 'T))

        do worker.ProgressChanged.Add(fun args ->
            progress.Trigger (args.ProgressPercentage,(args.UserState :?> 'T)))

        member x.WorkerCompleted  = completed.Publish
        member x.WorkerCancelled  = cancelled.Publish
        member x.WorkerError      = error.Publish
        member x.ProgressChanged  = progress.Publish

        // Delegate the remaining members to the underlying worker
        member x.RunWorkerAsync()    = worker.RunWorkerAsync()
        member x.CancelAsync()       = worker.CancelAsync()

    let fibOneStep (fibPrevPrev:bigint,fibPrev) = (fibPrev, fibPrevPrev+fibPrev)

    let worker = new IterativeBackgroundWorker<_>( fibOneStep,(1I,1I),100)

    worker.WorkerCompleted.Add(fun result ->
        MessageBox.Show(sprintf "Result = %A" result) |> ignore)
    worker.ProgressChanged.Add(fun (percentage, state) ->
        printfn "%d%% complete, state = %A" percentage state)

    worker.RunWorkerAsync()