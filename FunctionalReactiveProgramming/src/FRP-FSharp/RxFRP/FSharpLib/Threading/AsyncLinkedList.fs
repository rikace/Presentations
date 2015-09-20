namespace Easj360FSharp

open System
open System.Collections.Generic
open AgentHelper

//type Agent<'T> = MailboxProcessor<'T>

//type test() = 

//    let lst = new LinkedList<int>()
//    for i in [0..1000000] do ignore(lst.AddFirst(i))
//    for i in [0..1000000] do ignore(lst.AddLast(i))
//    for i in [0..1000000] do ignore(lst.RemoveFirst())
//    for i in [0..1000000] do ignore(lst.RemoveLast())
//    for i in [0..1000000] do ignore(lst.First)
//    for i in [0..1000000] do ignore(lst.Last)


type internal BlockingLinkedListMessage<'T> = 
  | Add of 'T * AsyncReplyChannel<unit> 
  | Get of Node * AsyncReplyChannel<'T>
and Node =
| First
| Last

type BlockingLinkedList<'T>(maxLength) =
  do 
    if maxLength <= 0 then 
      invalidArg "maxLenght" "Maximal length of the queue should be positive."

  [<VolatileField>]
  let mutable count = 0

  let agent = Agent.Start(fun agent ->
    let linkedList = new LinkedList<_>()
    let pending = new Queue<_>()

    let rec emptyQueue() = 
      agent.Scan(fun msg ->
        match msg with 
        | Add(value:'T, reply:AsyncReplyChannel<unit>) -> Some <| async {
            ignore <| linkedList.AddFirst(value)
            count <- linkedList.Count
            reply.Reply()
            return! nonEmptyQueue() }
        | _ -> None )
    
    and nonEmptyQueue() = async {
      let! msg = agent.Receive()
      match msg with 
      | Add(value:'T, reply:AsyncReplyChannel<unit>) -> 
          if linkedList.Count < maxLength then
            ignore <| linkedList.AddFirst(value)
            count <- linkedList.Count
            reply.Reply()
          else 
            pending.Enqueue(value, reply) 
          return! nonEmptyQueue()
      | Get(node, reply) -> 
          let item = match node with
                     | First    ->  let nodeExtracted = linkedList.First
                                    linkedList.RemoveFirst()          
                                    nodeExtracted
                     | Last     ->  let nodeExtracted = linkedList.Last
                                    linkedList.RemoveLast()
                                    nodeExtracted
          while linkedList.Count < maxLength && pending.Count > 0 do
            let (itm:'T, caller:AsyncReplyChannel<unit>) = pending.Dequeue()
            linkedList.AddFirst(itm) |>ignore
            caller.Reply()
          count <- linkedList.Count
          reply.Reply(item.Value)
          if linkedList.Count = 0 then return! emptyQueue()
          else return! nonEmptyQueue() }

    emptyQueue() )

  member x.Count = count
  member x.AsyncAdd(v:'T, ?timeout) = 
    agent.PostAndAsyncReply((fun ch -> Add(v, ch)), ?timeout=timeout)

  member x.AsyncGetFirst(?timeout) = 
    agent.PostAndAsyncReply((fun ch -> Get(First, ch)), ?timeout=timeout)
  
  member x.AsyncGetLast(?timeout) = 
    agent.PostAndAsyncReply((fun ch -> Get(Last, ch)), ?timeout=timeout)





/////////////////////////////////////////

type internal AsyncLinkedListMessage<'T> = 
  | Add of 'T 
  | Get of Node * AsyncReplyChannel<'T>
//and Node =
//| First
//| Last


type AsyncLinkedList<'T>() =
  
  [<VolatileField>]
  let mutable count = 0

  let agent = Agent.Start(fun agent ->    
    let rec loop(lnklst:LinkedList<_>, n:int) = async {
      let! msg = agent.Receive()
      match msg with 
      | Add(value:'T) ->    ignore <| lnklst.AddFirst(value)
                            count <- lnklst.Count
                            return! loop(lnklst, (n+1))
      | Get(node, reply) -> 
          let item = match node with
                     | First    ->  let nodeExtracted = lnklst.First
                                    lnklst.RemoveFirst()          
                                    nodeExtracted
                     | Last     ->  let nodeExtracted = lnklst.Last
                                    lnklst.RemoveLast()
                                    nodeExtracted
          count <- lnklst.Count
          reply.Reply(item.Value)
          return! loop(lnklst, (n-1)) }
    loop((new LinkedList<_>()),0))


  member x.Count = count
  member x.AsyncAdd(v:'T) = agent.Post(Add(v))

  member x.AsyncGetFirst() = agent.PostAndAsyncReply(fun ch -> Get(First, ch))
  
  member x.AsyncGetLast() = agent.PostAndAsyncReply(fun ch -> Get(Last, ch))

  (*
  [<RequireQualifiedAccess>]
module LinkedList =   
   open System.Collections.Generic
   let empty<'a> = LinkedList<'a>()
   let ofSeq<'a> (xs:'a seq) = LinkedList<'a>(xs)
   let find (f:'a->bool) (xs:LinkedList<'a>) =    
      let node = ref xs.First
      while !node <> null && not <| f((!node).Value) do 
         node := (!node).Next
      !node   
   let findi (f:int->'a->bool) (xs:LinkedList<'a>) =
      let node = ref xs.First
      let i = ref 0
      while !node <> null && not <| f (!i) (!node).Value do 
         incr i; node := (!node).Next
      if !node = null then -1 else !i
   let nth n (xs:LinkedList<'a>) =      
      if n >= xs.Count then 
        let message = "The input sequence has an insufficient number of elements."       
        raise <| new System.ArgumentException(message,paramName="n")
      let node = ref xs.First      
      for i = 1 to n do node := (!node).Next
      !node
  *)

//  let form = new System.Windows.Forms.Form(Visible = true, Width = 400, Height = 400)
//  let textBox = new System.Windows.Forms.TextBox(Dock = System.Windows.Forms.DockStyle.Fill, Font = new System.Drawing.Font("Lucida Comics", float32(12)), Multiline = true)
//  form.Controls.Add(textBox)
//
//  
//  let show(x) = textBox.Text <- sprintf "%A" x
//  let append(x) = textBox.AppendText( sprintf "%A\n" x )
//  
//  show "Riccardo"
//  append "Ciao"
//
//
//
//  let form = new System.Windows.Forms.Form(Visible = true, Width = 400, Height = 400)
//  let grid = new System.Windows.Forms.DataGrid(Dock = System.Windows.Forms.DockStyle.Fill)
//  form.Controls.Add(grid)
//
//  let bindData(x) = grid.DataSource <- x
//
//  let data = Array.init 30 (fun i -> sprintf "%d" i)
//  bindData(data)
                    

