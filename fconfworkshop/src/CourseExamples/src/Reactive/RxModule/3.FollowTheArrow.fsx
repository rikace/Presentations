#r "../../packages/Rx-Core.2.2.5/lib/net45/System.Reactive.Core.dll"
#r "../../packages/Rx-Interfaces.2.2.5/lib/net45/System.Reactive.Interfaces.dll"
#r "../../packages/Rx-Linq.2.2.5/lib/net45/System.Reactive.Linq.dll"
#r "../../packages/Rx-Xaml.2.2.5/lib/net45/System.Reactive.Windows.Threading.dll"
#r "WindowsBase.dll"
#r "PresentationCore.dll"
#r "PresentationFramework.dll"
#r "System.Xaml.dll"
#r "UIAutomationTypes.dll"
#load "Utils.fs"

open System
open System.Reactive
open System.Reactive.Linq
open System.Reactive.Subjects
open System.Threading
open System.Threading.Tasks
open System.Collections.Generic
open AgentModule
open System.Windows 
open System.Windows.Controls 
open System.Windows.Media 

// ===========================================
// Follow the mouse with RX
// ===========================================
    

type MyWindow() as this = 
    inherit Window()   

    let WIDTH = 20.0
    let canvas = new Canvas(Width=800.0, Height=400.0, Background = Brushes.White) 
    let chars = 
        "Reactive Extensions are awsome!"
        |> Seq.map (fun c -> 
            new TextBlock(Width=WIDTH, Height=30.0, FontSize=20.0, Text=string c, 
                            Foreground=Brushes.Black, Background=Brushes.White))
        |> Seq.toArray 
    do
        let title = sprintf "Mouse Move Sample - x = %d"
        this.Content <- canvas 
        this.Topmost <- true
        this.Title <- title 0
        this.SizeToContent <- SizeToContent.WidthAndHeight  
        for tb in chars do                                     
            canvas.Children.Add(tb) |> ignore 

        this.MouseMove 
        |> Observable.filter (fun ea -> ea.LeftButton = Input.MouseButtonState.Pressed 
                                        || ea.RightButton = Input.MouseButtonState.Pressed)
        |> Observable.map (fun ea -> ea.GetPosition(this))
        |> Observable.filter (fun p -> p.X < 300.0)
        |> Observable.add (fun p -> 
            async {
                this.Title <- title (int p.X)
                for i in 0..chars.Length-1 do
                    do! Async.Sleep(90)
                    Canvas.SetTop(chars.[i], p.Y)
                    Canvas.SetLeft(chars.[i], p.X + float i*WIDTH)
            } |> Async.StartImmediate 
            (*  Async.StartImmediate to start an asynchronous 
                computation on the current thread. Often, 
                an asynchronous operation needs to update UI, 
                which should always be done on the UI thread. 
                When your asynchronous operation needs to begin 
                by updating UI, Async.StartImmediate is a better choice *)
        ) 

[<System.STAThread()>] 
do  
    let app =  new Application() 
    app.Run(new MyWindow()) |> ignore 

