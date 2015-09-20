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

type MyWindow() as this = 
    inherit Window()   

    let WIDTH = 20.0
    let canvas = new Canvas(Width=800.0, Height=400.0, Background = Brushes.White) 
    let ball = new Ellipse(Width=50.0, Height=50.0, Fill=Brushes.Red)

    let moveBall(x, y) ball =
        let ball : Ellipse = ball
        Canvas.SetLeft(ball, x)
        Canvas.SetTop(ball, y)

    do
        let title = sprintf "Mouse Move Ball Sample - x = %d"
        this.Content <- canvas 
        this.Topmost <- true
        this.Title <- title 0
        this.SizeToContent <- SizeToContent.WidthAndHeight                          
        canvas.Children.Add(ball) |> ignore 

        this.MouseMove // Event-Combinators
        |> Observable.map (fun ea -> 
                let pos = ea.GetPosition(this)
                pos.X - 25.0, pos.Y - 25.0)
        |> Observable.filter (fun (x,y) ->   y > 100.0 && y < 250.0 && x > 100.0 && x < 350.0)
        |> Observable.add( fun point -> moveBall point ball)
            //} |> Async.StartImmediate 
            (*  Async.StartImmediate to start an asynchronous 
                computation on the current thread. Often, 
                an asynchronous operation needs to update UI, 
                which should always be done on the UI thread. 
                When your asynchronous operation needs to begin 
                by updating UI, Async.StartImmediate is a better choice *)
        

[<System.STAThread()>] 
do  
    let app =  new Application() 
    app.Run(new MyWindow()) |> ignore 


