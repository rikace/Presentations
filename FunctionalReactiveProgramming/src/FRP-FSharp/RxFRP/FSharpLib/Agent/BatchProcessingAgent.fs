namespace Easj360FSharp

open System
open System.Threading
open AgentHelper

// ----------------------------------------------------------------------------

/// Agent that can be used to implement batch processing. It creates groups
/// of messages (added using the Enqueue method) and emits them using the 
/// BatchProduced event. A group is produced when it reaches the maximal 
/// size or after the timeout elapses.

//let sync = SynchronizationContext.Current
//let proc = BatchProcessor<_>(10, sync)

type BatchProcessingAgent<'T>(bulkSize, timeout, ?eventContext:SynchronizationContext) = 

  let bulkEvent = new Event<'T[]>()

  let reportBatch batch =
    match eventContext with 
    | None -> 
        // No synchronization context - trigger as in the first case
        bulkEvent.Trigger(batch)
    | Some ctx ->
        // Use the 'Post' method of the context to trigger the event
        ctx.Post((fun _ -> bulkEvent.Trigger(batch)), null)


  let agent : Agent<'T> = Agent.Start(fun agent -> 
    let rec loop remainingTime messages = async {
      let start = DateTime.Now
      let! msg = agent.TryReceive(timeout = max 0 remainingTime)
      let elapsed = int (DateTime.Now - start).TotalMilliseconds
      match msg with 
      | Some(msg) when 
            List.length messages = bulkSize - 1 -> 
                reportBatch (msg :: messages |> List.rev |> Array.ofList)
                //bulkEvent.Trigger(msg :: messages |> List.rev |> Array.ofList)
                return! loop timeout []
      | Some(msg) ->
          return! loop (remainingTime - elapsed) (msg::messages)
      | None when List.length messages <> 0 -> 
            reportBatch (messages |> List.rev |> Array.ofList)
            //bulkEvent.Trigger(messages |> List.rev |> Array.ofList)
            return! loop timeout []
      | None -> 
          return! loop timeout [] }
    loop timeout [] )

  /// The event is triggered when a group of messages is collected. The
  /// group is not empty, but may not be of the specified maximal size
  /// (when the timeout elapses before enough messages is collected)
  [<CLIEvent>]
  member x.BatchProduced = bulkEvent.Publish

  /// Sends new message to the agent
  member x.Enqueue v = agent.Post(v)
  


//open System.Drawing
//open System.Windows.Forms
//
//let frm = new Form()
//let lbl = new Label(Font = new Font("Calibri", 20.0f), Dock = DockStyle.Fill)
//lbl.TextAlign <- ContentAlignment.MiddleCenter
//frm.Controls.Add(lbl)
//frm.Show()
//
//let ag = new BatchProcessingAgent<_>(5, 5000)
//frm.KeyPress.Add(fun e -> ag.Enqueue(e.KeyChar))
//ag.BatchProduced
//  |> Event.map (fun chars -> new String(chars))
//  |> Event.scan (+) ""
//  |> Event.add (fun str -> lbl.Text <- str)

module Template =
    /// Type alias that gives convenient name to F# agent type
    type Agent<'T> = MailboxProcessor<'T>

    /// Agent that implements batch processing
    type BatchProcessor<'T>(count) =
      // Event used to report aggregated batches to the user
      let batchEvent = new Event<'T[]>()
      // Trigger event on the thread where the agent is running
      let reportBatch batch =
        try
          // If the handler throws, we need to handle the exception
          batchEvent.Trigger(batch)
        with e ->
          printfn "Event handler failed: %A" e

      // Start an agent that implements the batching
      let agent = Agent<'T>.Start(fun inbox -> async {
        while true do
          // Repeatedly allocate a new queue 
          let queue = new ResizeArray<_>()
          // Add specified number of messages to the queue
          for i in 1 .. count do
            let! msg = inbox.Receive()
            queue.Add(msg)
          // Report the batch as an array
          reportBatch (queue.ToArray()) })

      /// Event that is triggered when a batch is collected
      member x.BatchProduced = batchEvent.Publish

      /// The method adds one object to the agent
      member x.Post(value) = agent.Post(value)

module ThreadPool =
// [snippet:Triggering events in a thread pool]    

    /// Agent that implements batch processing
    type BatchProcessor<'T>(count) =
      // Event used to report aggregated batches to the user
      let batchEvent = new Event<'T[]>()
      // Trigger event in a thread pool
      let reportBatch batch =
        // Create simple workflow & start it in the background
        async { batchEvent.Trigger(batch) } 
        |> Async.Start

      // Start an agent that implements the batching
      let agent = Agent<'T>.Start(fun inbox -> async {
        while true do
          // Repeatedly allocate a new queue 
          let queue = new ResizeArray<_>()
          // Add specified number of messages to the queue
          for i in 1 .. count do
            let! msg = inbox.Receive()
            queue.Add(msg)
          // Report the batch as an array
          reportBatch (queue.ToArray()) })

      /// Event that is triggered when a batch is collected
      member x.BatchProduced = batchEvent.Publish

      /// The method adds one object to the agent
      member x.Post(value) = agent.Post(value)

// [snippet:Reporting events using synchronization context]
open System.Threading

/// Agent that implements batch processing (eventContext can 
/// be provided to specify synchronization context for event reporting)
type BatchProcessor<'T>(count, ?eventContext:SynchronizationContext) =
  /// Event used to report aggregated batches to the user
  let batchEvent = new Event<'T[]>()

  /// Triggers event using the specified synchronization context
  /// (or directly, if no synchronization context is specified)
  let reportBatch batch =
    match eventContext with 
    | None -> 
        // No synchronization context - trigger as in the first case
        batchEvent.Trigger(batch)
    | Some ctx ->
        // Use the 'Post' method of the context to trigger the event
        ctx.Post((fun _ -> batchEvent.Trigger(batch)), null)

  // Start an agent that implements the batching
  let agent = Agent<'T>.Start(fun inbox -> async {
    while true do
      // Repeatedly allocate a new queue 
      let queue = new ResizeArray<_>()
      // Add specified number of messages to the queue
      for i in 1 .. count do
        let! msg = inbox.Receive()
        queue.Add(msg)
      // Report the batch as an array
      reportBatch (queue.ToArray()) })(*[/omit]*)

  /// Event that is triggered when a batch is collected
  member x.BatchProduced = batchEvent.Publish
  /// The method adds one object to the agent
  member x.Post(value) = agent.Post(value)

//// [snippet:Capturing current (user-interface) context]
//// Agent that will trigger events on the current (GUI) thread
//let sync = SynchronizationContext.Current
//let proc = BatchProcessor<_>(10, sync)
//
//// Start some background work that will report batches to GUI thread
//
//async {
//  for i in 0 .. 1000 do 
//    proc.Post(i) } |> Async.Start