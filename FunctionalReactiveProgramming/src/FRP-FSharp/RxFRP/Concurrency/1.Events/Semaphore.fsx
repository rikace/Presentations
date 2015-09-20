#load "..\CommonModule.fsx"
#r "WindowsBase.dll"
#r "PresentationCore.dll"
#r "PresentationFramework.dll"
#r "System.Xaml.dll"
#r "UIAutomationTypes.dll"

(*  every time the mouse moves, I want to have the first letter of the sentence
    move to follow it, and then after a short delay have the second letter move 
    to follow it, and then after a short delay the next letter, and so on.  
    Since this little loop-over-each-character runs asynchronously, we don’t jam up the UI *)
open System.Windows 
open System.Windows.Controls 
open System.Windows.Media 
open System.Windows.Shapes 
open System
open Common

type MyWindow() as this = 
    inherit Window()   

    let WIDTH = 20.0
    let canvas = new Canvas(Width=800.0, Height=400.0, Background = Brushes.White) 
    let green = new Ellipse(Width=60.0, Height=60.0, Fill=Brushes.Green)
    let orange = new Ellipse(Width=60.0, Height=60.0, Fill=Brushes.Orange)
    let red = new Ellipse(Width=60.0, Height=60.0, Fill=Brushes.Red)
    let gray = new Ellipse(Width=60.0, Height=60.0, Fill=Brushes.LightGray)
    let gray' = new Ellipse(Width=60.0, Height=60.0, Fill=Brushes.LightGray)
    let gray'' = new Ellipse(Width=60.0, Height=60.0, Fill=Brushes.LightGray)
    
    let display (current:Ellipse) =
        green.Visibility <- Visibility.Collapsed
        red.Visibility <- Visibility.Collapsed
        orange.Visibility <- Visibility.Collapsed
        current.Visibility <- Visibility.Visible 

    let semaphoreStates() = async {
        while true do
          let! md = Async.AwaitObservable(this.MouseLeftButtonDown)
          display(green) 
          let! md = Async.AwaitObservable(this.MouseLeftButtonDown)
          display(orange) 
          let! md = Async.AwaitObservable(this.MouseLeftButtonDown)
          display(red)  
        }

    do
        let title = sprintf "Mouse Move Ball Sample - x = %d"
        canvas.Children.Add(gray) |> ignore 
        canvas.Children.Add(gray') |> ignore 
        canvas.Children.Add(gray'') |> ignore 
        canvas.Children.Add(red) |> ignore 
        canvas.Children.Add(green) |> ignore 
        canvas.Children.Add(orange) |> ignore 

        Canvas.SetTop(red, 140.)
        Canvas.SetTop(green, 0.)
        Canvas.SetTop(orange, 70.)
        Canvas.SetTop(gray, 0.)
        Canvas.SetTop(gray', 70.)
        Canvas.SetTop(gray'', 140.)

        this.Content <- canvas 
        this.Topmost <- true
        this.Title <- title 0
        this.SizeToContent <- SizeToContent.WidthAndHeight                          
        
        semaphoreStates() |> Async.StartImmediate

[<System.STAThread()>] 
do  
    let app =  new Application() 
    app.Run(new MyWindow()) |> ignore 


