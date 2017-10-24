#r "WindowsBase.dll"
#r "PresentationCore.dll"
#r "PresentationFramework.dll"
#r "System.Xaml.dll"
#r "UIAutomationTypes.dll"
#load "Utils.fs"

open Utils
open System
open System.Windows 
open System.Windows.Shapes
open System.Windows.Media
open System.Windows.Controls

type Sempahore() as this = 
    inherit Window()   

    let canvas = new Canvas(Width=150.0, Height=300.0, Background = Brushes.White) 
    let elipseGray0 = Ellipse(Width=60., Height=60., Fill=Brushes.LightGray)    
    let elipseGray70 = Ellipse(Width=60., Height=60., Fill=Brushes.LightGray)
    let elipseGray140 = Ellipse(Width=60., Height=60., Fill=Brushes.LightGray)
    let elipseGreen = Ellipse(Width=60., Height=60., Fill=Brushes.Green)    
    let elipseOrange = Ellipse(Width=60., Height=60., Fill=Brushes.Orange)
    let elipseRed = Ellipse(Width=60., Height=60., Fill=Brushes.Red)

    let display (current:Ellipse) =
        elipseGreen.Visibility <- Visibility.Collapsed
        elipseRed.Visibility <- Visibility.Collapsed
        elipseOrange.Visibility <- Visibility.Collapsed
        current.Visibility <- Visibility.Visible 
    

    let semaphoreStates() = 
        let rec loop() = async {
            for ellipse in [elipseGreen; elipseOrange; elipseRed] do
                let! _ = Async.AwaitObservable(this.MouseLeftButtonDown)
                display(ellipse) 
            return! loop()
        }
        loop()


    do
        this.Width <- 150.
        this.Height <- 300.
        canvas.Children.Add(elipseGray0) |> ignore
        canvas.Children.Add(elipseGray70) |> ignore
        canvas.Children.Add(elipseGray140) |> ignore
        canvas.Children.Add(elipseGreen) |> ignore
        canvas.Children.Add(elipseOrange) |> ignore
        canvas.Children.Add(elipseRed) |> ignore
        
        Canvas.SetTop(elipseGray0, 0.)
        Canvas.SetTop(elipseGray70, 70.)
        Canvas.SetTop(elipseGray140, 140.)
        Canvas.SetTop(elipseGreen, 0.)
        Canvas.SetTop(elipseOrange, 70.)
        Canvas.SetTop(elipseRed, 140.)
        this.Content <- canvas

        
        semaphoreStates() |> Async.StartImmediate

[<System.STAThread()>] 
do  let app =  new Application() 
    app.Run(new Sempahore()) |> ignore 
