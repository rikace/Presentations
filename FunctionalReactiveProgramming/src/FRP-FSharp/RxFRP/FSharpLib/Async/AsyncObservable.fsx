#r "System.Windows.Forms.dll"
#r "WindowsBase.dll"
#r "PresentationCore.dll"
#r "PresentationFramework.dll"

open System.Windows 
open System.Windows.Controls 
open System.Windows.Media 

type MyWindow() as this = 
    inherit Window()   

    let WIDTH = 20.0
    let canvas = new Canvas(Width=800.0, Height=400.0, Background = Brushes.White) 
    let chars = 
        " F# reacts to events!"
        |> Seq.map (fun c -> 
            new TextBlock(Width=WIDTH, Height=30.0, FontSize=20.0, Text=string c, 
                          Foreground=Brushes.Black, Background=Brushes.White))
        |> Seq.toArray 
    do
        this.Content <- canvas 
        this.Title <- "MyWPFWindow" 
        this.SizeToContent <- SizeToContent.WidthAndHeight  
        for tb in chars do                                     
            canvas.Children.Add(tb) |> ignore 

        this.MouseMove 
        |> Observable.map (fun ea -> ea.GetPosition(this))
        //|> Observable.filter (fun p -> p.X < 300.0)
        |> Observable.add (fun p -> 
            async {
                for i in 0..chars.Length-1 do
                    do! Async.Sleep(90)
                    Canvas.SetTop(chars.[i], p.Y)
                    Canvas.SetLeft(chars.[i], p.X + float i*WIDTH)
            } |> Async.StartImmediate
        )

[<System.STAThread()>] 
do  
    let app =  new Application() 
    app.Run(new MyWindow()) |> ignore 
