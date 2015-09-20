module MailBoxScanState

type Message = 
  | ModifyState of int
  | Block
  | Resume

let cancel = new System.Threading.CancellationTokenSource()

// Listing Mailbox processor using state machine
  
let mbox = MailboxProcessor.Start((fun mbox ->
    // Represents the blocked state
    let rec blocked(n) = 
      printfn "Blocking"
      // Only process the 'Resume' message
      mbox.Scan(fun msg ->
        match msg with
        // Return workflow to continue with
        | Resume -> Some(async {
            printfn "Resuming"
            return! processing(n) })
        // Other messages cannot be processed now
        | _ -> None)
        
    // Represents the active  state
    and processing(n) = async {
      printfn "Processing: %d" n
      // Process any message
      let! msg = mbox.Receive()
      match msg with
      | ModifyState(by) -> return! processing(n + by)
      | Resume -> return! processing(n)
      | Block -> return! blocked(n) }
    processing(0)  ), cancel.Token)
  
// Listing Sending messages from multiple threads 

open System
open System.Threading
  
// Thread performing calculations
let modifyThread() =
  let rnd = new Random(Thread.CurrentThread.ManagedThreadId)  
  while true do
    Thread.Sleep(500)
    // Send an update to the mailbox
    mbox.Post(ModifyState(rnd.Next(11) - 5)) 

let blockThread() =
  while true do
    Thread.Sleep(2000)
    mbox.Post(Block)    
    // Block the processing for one and half seconds
    Thread.Sleep(1500)
    mbox.Post(Resume) 


for proc in [ blockThread; modifyThread; modifyThread ] do
  Async.Start(async { proc() }, cancel.Token)

cancel.Cancel()