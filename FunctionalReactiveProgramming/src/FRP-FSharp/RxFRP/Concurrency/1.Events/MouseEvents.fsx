#r "WindowsBase.dll"
#r "PresentationCore.dll"
#r "PresentationFramework.dll"
#r "System.Xaml.dll"
#r "UIAutomationTypes.dll"


open System.Windows 
open System.Windows.Controls 
open System.Windows.Media 
open System.Windows.Shapes 
open System.Windows.Media

let (?) (this : Control) (prop : string) : 'T =
  this.FindName(prop) :?> 'T

type MyWindow() as this = 
    inherit Window()   

    let WIDTH = 20.0
    let canvas = new Canvas(Width=800.0, Height=400.0, Background = Brushes.White) 
    let ball = new Ellipse(Name="Ball", Width=50., Height=50., Fill=System.Windows.Media.Brushes.SaddleBrown)

    let moveBall(x, y) =
        let ball : Ellipse = this?Ball
        Canvas.SetLeft(ball, x)
        Canvas.SetTop(ball, y)

    do
        let title = sprintf "Mouse Move Sample - x = %d"
        this.Content <- canvas 
        this.Topmost <- true
        this.Title <- title 0
        this.SizeToContent <- SizeToContent.WidthAndHeight  
        canvas.Children.Add(ball) |> ignore 

        this.MouseMove
        |> Observable.map (fun me -> 
              let pos = me.GetPosition(this)
              pos.X - 25.0, pos.Y - 25.0 )
        |> Observable.filter (fun (x, y) -> 
              y > 100.0 && y < 250.0 && x > 100.0 && x < 350.0 )
        |> Observable.add moveBall
            

[<System.STAThread()>] 
do  
    let app =  new Application() 
    app.Run(new MyWindow()) |> ignore 