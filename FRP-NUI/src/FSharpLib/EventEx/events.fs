namespace global

module Event = 
    /// An event which triggers on every 'n' triggers of the input event
    let every n (ev:IEvent<_>) = 
        let out = new Event<_>()
        let count = ref 0 
        ev.Add (fun arg -> incr count; if !count % n = 0 then out.Trigger arg)
        out.Publish

    /// An event which triggers on every 'n' triggers of the input event
    let window n (ev:IEvent<_>) = 
        let out = new Event<_>()
        let queue = System.Collections.Generic.Queue<_>()
        ev.Add (fun arg -> queue.Enqueue arg; 
                           if queue.Count >= n then 
                                out.Trigger (queue.ToArray()); 
                                queue.Dequeue() |> ignore)
        out.Publish

    let pairwise  (ev:IEvent<_>) = 
        let out = new Event<_>()
        let queue = System.Collections.Generic.Queue<_>()
        ev.Add (fun arg -> queue.Enqueue arg; 
                           if queue.Count >= 2 then 
                                let elems = queue.ToArray()
                                out.Trigger (elems.[0], elems.[1])
                                queue.Dequeue() |> ignore)
        out.Publish

    let toObservableCollection (e:IEvent<_,_>) = 
        let oc = new System.Collections.ObjectModel.ObservableCollection<'T>()
        e.Add(fun x -> oc.Add x) 
        oc

module Seq = 
  let mutable ct = new System.Threading.CancellationTokenSource()
  let cancelPrevious() = ct.Cancel(); ct <- new System.Threading.CancellationTokenSource(); ct.Token
  let startAsUniqueEvent (s:seq<_>) = 
      let ctxt = System.Threading.SynchronizationContext.Current
      if ctxt = null then invalidOp "This function may only be called from a thread where SynchronizationContext.Current is not null"
      let oc = new System.Collections.ObjectModel.ObservableCollection<'T>()
      let ev = new Event<_>()
      let job = async { for x in s do ctxt.Post((fun _ -> ev.Trigger x),null) } 
      Async.Start(job,cancelPrevious())
      ev.Publish

