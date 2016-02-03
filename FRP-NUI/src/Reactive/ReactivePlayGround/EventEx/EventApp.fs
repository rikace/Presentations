namespace Easj360FSharp 

open System
open System.Drawing
open System.Windows
open System.Windows.Shapes 
open System.Windows.Media
open System.Windows.Controls
open EventObservable

module EventApp =
 
 let (?) (this : Control) (prop : string) : 'T =
  this.FindName(prop) :?> 'T

 
 type Semaphore() =
  
  let form = new System.Windows.Forms.Form()
  let ev = form.MouseClick
  
  let red = System.Windows.Media.Colors.Red
  let green = System.Windows.Media.Colors.Green
  let orange = System.Windows.Media.Colors.Orange

  
  let display (current:System.Windows.Media.Color) =
    printfn "%d" (current.A + current.B + current.G)

  let semaphoreStates() = 
    async {
    while true do
      let! md = Async.AwaitObservable(ev)
      display(green) 
      let! md = Async.AwaitObservable(ev)
      display(orange) 
      let! md = Async.AwaitObservable(ev)
      display(red)  
    }

  do
    semaphoreStates() |> Async.StartImmediate

  let main = new Window()
  
  let moveControl (ctl:FrameworkElement) (start:Point) (finish:Point) =
    ctl.Width <- abs(finish.X - start.X)
    ctl.Height <- abs(finish.Y - start.Y)
    Canvas.SetLeft(ctl, min start.X finish.X)
    Canvas.SetTop(ctl, min start.Y finish.Y)
  
  let transparentGray = 
    SolidColorBrush(Color.FromArgb(128uy, 164uy, 164uy, 164uy))

  let rec waiting() = async {
    let! md = Async.AwaitObservable(ev)
    let rc = new Canvas(Background = transparentGray)
    do! drawing(rc, md.Location) }

  and drawing(rc:Canvas, pos) = async {
    let! evt = Async.AwaitObservable(main.MouseLeftButtonUp, main.MouseMove)
    match evt with
    | Choice1Of2(up) -> 
        rc.Background <- SolidColorBrush()
        do! waiting() 
    | Choice2Of2(move) ->
       // moveControl rc pos (move.GetPosition(main))
        do! drawing(rc, pos) }

  do
    waiting() |> Async.StartImmediate


#if INTERACTIVE
#r "WindowsBase.dll"
#r "PresentationCore.dll"
#r "PresentationFramework.dll"
#r "System.Xaml.dll"
#endif

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
//
//[<System.STAThread()>] 
//do  
//    let app =  new Application() 
//    app.Run(new MyWindow()) |> ignore 