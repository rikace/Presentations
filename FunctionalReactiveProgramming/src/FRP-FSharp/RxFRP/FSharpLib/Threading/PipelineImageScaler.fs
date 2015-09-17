namespace Easj360FSharo

open System
open System.Collections.Concurrent
open System.Collections.Generic
open System.Diagnostics.CodeAnalysis
open System.Drawing
open System.IO
open System.Linq
open System.Threading
open System.Threading.Tasks

module PipelineImageScaler =

/// A convenience type alias for 'MailboxProcessor<T>' type
type Agent<'T> = MailboxProcessor<'T>

/// Type of messages internally used by 'BlockingQueueAgent<T>'
type internal BlockingAgentMessage<'T> = 
  // Send item to the queue and block until it is added 
  | Add of 'T * AsyncReplyChannel<unit> 
  // Get item from the queue (block if no item available)
  | Get of AsyncReplyChannel<'T>


/// <summary> Agent that implements an asynchronous blocking queue. </summary>
/// <remarks>
///   The queue has maximal length (maxLength) and if the queue is 
///   full, adding to it will (asynchronously) block the caller. When
///   the queue is empty, the caller will be (asynchronously) blocked
///   unitl an item is available.
/// </remarks>
type BlockingQueueAgent<'T>(maxLength) =
  do 
    if maxLength <= 0 then 
      invalidArg "maxLenght" "Maximal length of the queue should be positive."

  // We keep the number of elements in the queue in a local field to
  // make it immediately available (so that 'Count' property doesn't 
  // have to use messages - which would be a bit slower)
  [<VolatileField>]
  let mutable count = 0

  let agent = Agent.Start(fun agent ->
    // Keeps a list of items that are in the queue
    let queue = new Queue<_>()
    // Keeps a list of blocked callers and additional values
    let pending = new Queue<_>()

    // If the queue is empty, we cannot handle 'Get' message
    let rec emptyQueue() = 
      agent.Scan(fun msg ->
        match msg with 
        | Add(value, reply) -> Some <| async {
            queue.Enqueue(value)
            count <- queue.Count
            reply.Reply()
            return! nonEmptyQueue() }
        | _ -> None )

    // If the queue is non-empty, we can handle all messages
    and nonEmptyQueue() = async {
      let! msg = agent.Receive()
      match msg with 
      | Add(value, reply) -> 
          // If the queue has space, we add item and notif caller back;
          // if it is full, we enqueue the request (and block caller)
          if queue.Count < maxLength then 
            queue.Enqueue(value)
            count <- queue.Count
            reply.Reply()
          else 
            pending.Enqueue(value, reply) 
          return! nonEmptyQueue()
      | Get(reply) -> 
          let item = queue.Dequeue()
          // We took item from the queue - check if there are any blocked callers
          // and values that were not added to the queue because it was full
          while queue.Count < maxLength && pending.Count > 0 do
            let itm, caller = pending.Dequeue()
            queue.Enqueue(itm)
            caller.Reply()
          count <- queue.Count
          reply.Reply(item)
          // If the queue is empty then switch the state, otherwise loop
          if queue.Count = 0 then return! emptyQueue()
          else return! nonEmptyQueue() }

    // Start with an empty queue
    emptyQueue() )

  /// Returns the number of items in the queue (immediately)
  /// (excluding items that are being added by callers that have been
  /// blocked because the queue was full)
  member x.Count = count

  /// Asynchronously adds item to the queue. The operation ends when
  /// there is a place for the item. If the queue is full, the operation
  /// will block until some items are removed.
  member x.AsyncAdd(v:'T, ?timeout) = 
    agent.PostAndAsyncReply((fun ch -> Add(v, ch)), ?timeout=timeout)

  /// Asynchronously gets item from the queue. If there are no items
  /// in the queue, the operation will block unitl items are added.
  member x.AsyncGet(?timeout) = 
    agent.PostAndAsyncReply(Get, ?timeout=timeout)

let disposeOnException f (obj:#IDisposable) =
    try f obj
    with _ -> 
      obj.Dispose()  
      reraise()

let runMessagePassing fileNames sourceDir queueLength displayFn (cts:CancellationTokenSource) = 

  let loadedImages = new BlockingQueueAgent<_>(queueLength)
  let scaledImages = new BlockingQueueAgent<_>(queueLength)    
  let filteredImages = new BlockingQueueAgent<_>(queueLength)    

  let loadImages = async {
    let clockOffset = Environment.TickCount
    let rec numbers n = seq { yield n; yield! numbers (n + 1) }
    for count, img in fileNames |> Seq.zip (numbers 0) do
      let info = "" //loadImage img sourceDir count clockOffset
      do! loadedImages.AsyncAdd(info) }

  let scalePipelinedImages = async {
    while true do 
      let! info = loadedImages.AsyncGet()
      //scaleImage info
      do! scaledImages.AsyncAdd(info) }

  // Image pipeline phase 3: Filter images (give them a 
  // speckled appearance by adding Gaussian noise)
  let filterPipelinedImages = async {
    while true do 
      let! info = scaledImages.AsyncGet()
      //filterImage info
      do! filteredImages.AsyncAdd(info) }

  // Image pipeline phase 4: Invoke the user-provided callback 
  // function (for example, to display the result in a UI)
  let displayPipelinedImages = 
    let rec loop count duration = async {
      let! info = filteredImages.AsyncGet()
      let displayStart = Environment.TickCount
//      info.QueueCount1 <- loadedImages.Count
//      info.QueueCount2 <- scaledImages.Count
//      info.QueueCount3 <- filteredImages.Count
//      displayImage info (count + 1) displayFn duration
      return! loop (count + 1) (Environment.TickCount - displayStart) }
    loop 0 0

  Async.Start(loadImages, cts.Token)
  Async.Start(scalePipelinedImages, cts.Token)
  Async.Start(filterPipelinedImages, cts.Token)

  try Async.RunSynchronously(displayPipelinedImages, cancellationToken = cts.Token)
  with :? OperationCanceledException -> () 
