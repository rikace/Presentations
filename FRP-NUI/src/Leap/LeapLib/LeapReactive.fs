module LeapReactive

open System
open System.Collections.Generic
open System.Threading
open System.Reactive
open System.Reactive.Linq
open System.Reactive.Concurrency
open System.Reactive.Disposables
open RxFsharp
open Leap
open System.Reactive
open System.Reactive.Subjects

type ReactiveListener() =
    inherit Leap.Listener()

    let frameObservable = new Subject<Frame>()

    override x.OnFrame(ctrl:Controller) =
        frameObservable.OnNext(ctrl.Frame())            

    member x.Frames() = 
        frameObservable :> IObservable<Frame>

    member x.RegisterObserver(observer:IObserver<Frame>) =
        frameObservable.Subscribe(observer)


    member x. FingersMoves() : IObservable<IGroupedObservable<Finger, Finger>> =
        x.Frames()
        |> Observable.flatmapSeq(fun f -> seq { for finger in f.Fingers -> finger })
        |> Observable.groupByUntil(fun f -> f) (fun f -> f.Throttle(TimeSpan.FromMilliseconds(300.)))
        |> Observable.take 1

    member x. FramePerSecond() =
        x.Frames().Timestamp().Buffer(2)
        |> Observable.subscribe(fun o -> 
            let seconds = (o.[1].Timestamp - o.[0].Timestamp).TotalSeconds
            let fps = int(1.0 / seconds) 
            printfn "Frame per second %d" fps)
        


type ReactiveListenerArbirary(f:Controller -> unit) =
    inherit Leap.Listener()
    override x.OnFrame(ctrl:Controller) = f ctrl



type LeapMotionReactive() =
    let ctx = System.Threading.SynchronizationContext.Current
    let post f =
        match ctx with
        | null ->  f()
        | _ -> ctx.Post((fun _ -> f()),null)

    let event = Event<_>()
    let reactiveListener = new ReactiveListenerArbirary(fun ctrl -> 
                    let f = ctrl.Frame()
                    post(fun () -> event.Trigger(f)))
    let run = new Leap.Controller(reactiveListener)
    member this.EventFrame = event.Publish


//let lp = new LeapMotionReactive()
//lp.EventFrame
//|> Event.filter(fun x -> x.Fingers.Count > 0)
//|> Event.map(fun x -> x.Fingers.[0].TipVelocity.Magnitude)
//|> Event.scan max
//|> Event.add(fun x -> printfn "Max vel = %A" x)
