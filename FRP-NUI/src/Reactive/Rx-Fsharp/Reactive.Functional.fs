#nowarn "40"
namespace Reactive.Functional

open System
open System.Windows.Forms

type HAction<'a> = delegate of 'a -> unit
type FreeableEvent<'a> = 
  abstract AddHandler : HAction<'a> -> unit
  abstract RemoveHandler : HAction<'a> -> unit
  abstract Stop : unit -> unit 

[<AutoOpenAttribute>]
module Reactive =
  let cns a _ = a

  type private Closure<'a>(f) =
    member x.Invoke(sender:obj, a:'a) : unit = f(a)
  
  let private invokeMethod = 
    let ty = typeof<Closure<_>>
    ty.GetMethod("Invoke", System.Reflection.BindingFlags.Public ||| System.Reflection.BindingFlags.Instance)
    
  let private closure f = 
    new Closure<_>(f)
    
  let private hndlRegister (e:FreeableEvent<_>) f = 
    let h = new HAction<_>(f)
    e.AddHandler(h)
    h  

  let private create (free) =
    let handlers = new ResizeArray<HAction<'a>>()
    (fun (v:'a) ->  for h in handlers.ToArray() do h.Invoke(v)),
    { new FreeableEvent<'a> with
      member x.AddHandler(h) = handlers.Add(h)
      member x.RemoveHandler(h) = 
        handlers.Remove(h) |> ignore
        if (handlers.Count = 0) then
          free() 
      member x.Stop() =
        handlers.Clear()
        free() }
  
  let public freeable (ev:IEvent<'del, 'a>) =
    let rec evt = create (fun() -> ev.RemoveHandler(del))
    and obj = evt |> fst |> closure<'a>
    and del = Delegate.CreateDelegate(typeof<'del>, obj, "Invoke") :?> 'del
    ev.AddHandler(del)
    evt |> snd

  // Standard operators

  let public first (ev:FreeableEvent<'a>) =
    let first = ref true
    let rec evt = create (fun () -> ev.RemoveHandler(h))
    and h = hndlRegister ev (fun arg -> 
      if (!first) then
        first := true
        ev.RemoveHandler(h)
        (fst evt) arg )
    evt |> snd
  
  let public map f (ev:FreeableEvent<'a>) =
    let rec evt = create (fun () -> ev.RemoveHandler(h))
    and h = hndlRegister ev (fun arg -> (fst evt) (f arg))
    evt |> snd

  let public filter f (ev:FreeableEvent<'a>) =
    let rec evt = create (fun () -> ev.RemoveHandler(h))
    and h = hndlRegister ev (fun arg -> if (f arg) then (fst evt) arg)
    evt |> snd

  let public fold f seed (ev:FreeableEvent<'a>) =
    let state = ref seed
    let rec evt = create (fun () -> ev.RemoveHandler(h))
    and h = hndlRegister ev (fun arg -> state := (f !state arg); (fst evt) !state)
    evt |> snd

  let public listen f (ev:FreeableEvent<'a>) =
    hndlRegister ev f |> ignore

  let public pass f (ev:FreeableEvent<'a>) =
    let rec evt = create (fun () -> ev.RemoveHandler(h) )
    and h = hndlRegister ev (fun arg -> f arg; (fst evt) arg)
    evt |> snd

  let public switchRecursive switchFunc (ev:FreeableEvent<_>) =
    let current = ref id
    let rec setupSwitch result ev = 
      let rec h = hndlRegister ev (fun arg ->
        (result |> fst) arg
        let nevt = switchFunc arg
        ev.RemoveHandler(h)
        setupSwitch result nevt )
      current := (fun () -> ev.RemoveHandler(h))
    let result = create (fun () -> (!current) ())
    setupSwitch result ev
    result |> snd

  let public sum ev = ev |> fold (+) 0
  let public sumBy f ev = ev |> fold (fun st a -> st + (f(a))) 0
  
  // Generators
  open System.Windows.Forms
  
  let public Repeatedly(interval) =
    let tmr = new System.Timers.Timer(interval)
    let rec evt = create (fun () -> 
      tmr.Enabled <- false
      tmr.Dispose() )
    tmr.Elapsed.Add(fun e -> 
        if (Application.OpenForms.Count > 0) then
          Application.OpenForms.[0].Invoke(new Action(fun () -> (fst evt) e.SignalTime)) |> ignore
        else
          (fst evt) e.SignalTime)
    tmr.Enabled <- true
    evt |> snd

  let public Never() =
    let evt = create (fun () -> (* *) () )
    evt |> snd

  let public After(interval, value) =
    Repeatedly(interval) |> map (cns value) |> first

  let public Merge(a:FreeableEvent<_>, b:FreeableEvent<_>) =
    let rec evt = create(fun () ->
      a.RemoveHandler(h)
      b.RemoveHandler(h) )
    and h = new HAction<_>(evt |> fst)
    a.AddHandler(h)
    b.AddHandler(h)
    evt |> snd
