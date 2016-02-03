namespace FollowArrowRX

open System.Windows
open System.Windows.Controls
open System.Windows.Media
open System.Reactive
open System.Reactive.Subjects
open System.Reactive.Linq
open System
open System.Threading
open Leap
open RxFsharp
open Microsoft.FSharp.Control

type HandStatus = 
    { FingersCount : int
      Coordinate : Point }
    static member Empty = 
        { FingersCount = 0
          Coordinate = Point(0., 0.) }

type ReactiveListener(windowWidth : float, windowHeight : float, ctx : SynchronizationContext) = 
    inherit Leap.Listener()

    let event = Event<HandStatus>()
    
    let post f = 
        match ctx with
        | null -> f()
        | _ -> ctx.Post((fun _ -> f()), null)
    
    let leapAgent = 
        MailboxProcessor.Start(fun inbox -> 
            let rec loop() = 
                async { 
                    let! (ctrl : Controller) = inbox.Receive()
                    // right hand only one finger move
                    use frame = ctrl.Frame()
                    let hand = frame.Hands.Rightmost
                    // 2D drawing coordinate systems put the origin at the top, left corner of the window
                    // naturally don’t use a z-axis. 
                    // this code maps Leap Motion coordinates to such a 2D system 
                    let finger = hand.Fingers.[0].TipPosition
                    let iBox = frame.InteractionBox
                    let normalizedPoint = iBox.NormalizePoint(finger, true)
                    let X = float normalizedPoint.x * windowWidth
                    let Y = (1. - float normalizedPoint.y) * windowHeight
                    post (fun () -> 
                        event.Trigger({ FingersCount = hand.Fingers.Count
                                        Coordinate = Point(X, Y) }))
                    return! loop()
                }
            loop())
    
    member x.OnPosition = event.Publish

    override x.OnFrame(ctrl : Controller) = leapAgent.Post(ctrl)

type FollowArrowListener(mainWindow : Window, chars : TextBlock []) = 
    let windowWidth = mainWindow.Width
    let windowHeight = mainWindow.Height
    let ctx = SynchronizationContext.Current
  
    let reactiveListener = new ReactiveListener(windowWidth, windowHeight, ctx)
    let leapCtrl = new Leap.Controller()
    do 
        leapCtrl.AddListener(reactiveListener) |> ignore
    
    member x.Start(window : Window) = 
        let (handOpen:IObservable<HandStatus>), (handClose:IObservable<HandStatus>) =
            reactiveListener.OnPosition 
            |> Observable.partition (fun p -> p.FingersCount > 0)
      
        handClose
        |> Observable.delay (TimeSpan.FromSeconds(1.))
        |> Observable.observeOnContext (ctx)
        |> Observable.scan (fun color (p : HandStatus) -> 
               if color = Brushes.Red then Brushes.Black
               else Brushes.Red) Brushes.Black
        |> Observable.add (fun p -> 
               for i in 0..chars.Length - 1 do
                   chars.[i].Foreground <- p)
      
        handOpen
        |> Observable.filter (fun p -> p.FingersCount = 1)
        |> Observable.add (fun p -> 
               async { 
                   let coordinate = p.Coordinate
                   window.Title <- sprintf "Mouse Postion x %f - y %f" coordinate.X coordinate.Y
                   for i in 0..chars.Length - 1 do
                       do! Async.Sleep(90)
                       Canvas.SetTop(chars.[i], coordinate.Y)
                       Canvas.SetLeft(chars.[i], coordinate.X + float i * 20.)
               }
               |> Async.StartImmediate)
      
        handOpen
        |> Observable.filter (fun p -> p.FingersCount = 2)
        |> Observable.scan (fun color p -> 
               if color = Brushes.White then Brushes.LightBlue
               else Brushes.White) Brushes.White
        |> Observable.add (fun p -> async { window.Background <- p } |> Async.StartImmediate)
    
    interface System.IDisposable with
        member x.Dispose() = 
            leapCtrl.RemoveListener(reactiveListener) |> ignore
            reactiveListener.Dispose()
            leapCtrl.Dispose()
