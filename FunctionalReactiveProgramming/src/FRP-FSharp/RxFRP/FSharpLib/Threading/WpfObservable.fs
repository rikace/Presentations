namespace Threading
//#light
//
//#r "WindowsBase"
//#r "PresentationCore"
//#r "PresentationFramework"

open System.Windows
open System.Windows.Controls
open System.Windows.Media 

module WpfEvent = 

    let f= new Window()
    let WIDTH = 20.0
    let canvas = new Canvas(Width=800.0, Height=400.0, Background = Brushes.White)
    let chars =
        "Ciao bella Bugghina !!"
        |> Seq.map (fun c ->
            new TextBlock(Width=WIDTH, Height=30.0, FontSize=20.0, Text=string c,
                          Foreground=Brushes.Red, Background=Brushes.White))
        |> Seq.toArray
    f.Content <- canvas
    f.Title <- "My Bugghina Window"
    f.SizeToContent <- SizeToContent.WidthAndHeight
    for tb in chars do
        canvas.Children.Add(tb) |> ignore 

    f.MouseMove
    |> Observable.map (fun ea -> ea.GetPosition(f))
    //|> Observable.filter (fun p -> p.X < 300.0)
    |> Observable.add (fun p ->
        async {
            for i in 0..chars.Length-1 do
                do! Async.Sleep(90)
                Canvas.SetTop(chars.[i], p.Y)
                Canvas.SetLeft(chars.[i], p.X + float i*WIDTH)
        } |> Async.StartImmediate
    )

    f.Show()
