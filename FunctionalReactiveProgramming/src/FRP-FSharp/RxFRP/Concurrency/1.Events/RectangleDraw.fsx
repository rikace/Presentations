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
open Common

type MyWindow() as this = 
    inherit Window()   

    let WIDTH = 20.0
    let canvas = new Canvas(Width=800.0, Height=400.0, Background = Brushes.White) 

    let moveControl (ctl:FrameworkElement) (start:Point) (finish:Point) =
        ctl.Width <- abs(finish.X - start.X)
        ctl.Height <- abs(finish.Y - start.Y)
        Canvas.SetLeft(ctl, min start.X finish.X)
        Canvas.SetTop(ctl, min start.Y finish.Y)


    let transparentGray = 
        SolidColorBrush(Color.FromArgb(128uy, 164uy, 164uy, 164uy))

    let rec waiting() = async {
        let! md = Async.AwaitObservable(canvas.MouseLeftButtonDown)
        let rc = new Canvas(Background = transparentGray)
        canvas.Children.Add(rc) |> ignore
        do! drawing(rc, md.GetPosition(canvas)) }

      and drawing(rc:Canvas, pos) = async {
        let! evt = Async.AwaitObservable(canvas.MouseLeftButtonUp, canvas.MouseMove)
        match evt with
        | Choice1Of2(up) -> 
            rc.Background <- SolidColorBrush(Colors.Red)
            do! waiting() 
        | Choice2Of2(move) ->
            moveControl rc pos (move.GetPosition(canvas))
            do! drawing(rc, pos) }


    do
        let title = sprintf "Drawing Recatngles"
        this.Content <- canvas 
        this.Topmost <- true
        this.SizeToContent <- SizeToContent.WidthAndHeight                          


        waiting() |> Async.StartImmediate
        

[<System.STAThread()>] 
do  
    let app =  new Application() 
    app.Run(new MyWindow()) |> ignore 



