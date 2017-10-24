open System
open Utils
open Computation
open System.Windows
open System.Windows.Controls
open System.Windows.Media
open System.Threading
open System.Windows.Threading
open System.Windows.Media.Imaging
open Microsoft.FSharp.NativeInterop

// Convert a BitmapSource to int array
let getPixels(img:BitmapSource) =    
    let stride = img.PixelWidth * 4
    let size = img.PixelHeight * stride
    let pixels = Array.zeroCreate<int> size
    img.CopyPixels(pixels, stride, 0)    
    pixels    

type ZoomControl(width:int,height:int) as self =
    inherit Window()

    let runInParallel = true
    
    // Creating controls
    let bitmap = WriteableBitmap(width,height, 96., 96., PixelFormats.Bgr32, null)
    let canvas = Canvas()
   
    // Set the initial Points for the fractal
    let mutable points = (-2.0, -1.0, 1.0, 1.0)
        
    let transparentGray = 
        SolidColorBrush(Color.FromArgb(128uy, 164uy, 164uy, 164uy))
    do  
        // Settings window size and content
        self.Width <- float width
        self.Height <- float height
        canvas.Children.Add(Image(Source=bitmap)) |> ignore
        self.Content <- canvas 

    // Render the Fractal
    let render (x1,y1,x2,y2) (offset, height, stripe) (pixels:int array)  =   
        let dx, dy = x2-x1, y2-y1
        for y = 0 to height-1 do
            for x = 0 to width-1 do
                let y = offset + (y*stripe)
                pixels.[x+y*width] <-
                    let x = ((float x/float width) * dx) + x1
                    let y = ((float y/float (height*stripe)) * dy) + y1 
                    match Complex.Create(x, y) with
                    | DidNotEscape -> 0xff000000
                    | Escaped i -> 
                        let i = 255 - i
                        0xff000000 + (i <<< 16) +  (i <<< 8) + i
        // Update the UI using the correct Thread
        ignore <| self.Dispatcher.Invoke(Action(fun () -> 
            bitmap.WritePixels(new Int32Rect(0, 0, width,(height * (offset + 1))), pixels, width * 4, offset)))

    
    do render points (0,height,1) (getPixels(bitmap))
            
    // Copy the selected portion of the image into a new WriteableBitmap 
    let copy (l,t,w,h) =
        let selection =  WriteableBitmap(int w, int h, 96., 96., PixelFormats.Bgr32, null)
        let source = getPixels bitmap 
        for y = 0 to int h - 1 do
            for x = 0 to int w - 1 do
                let c = source.[int l + x + ((int t + y) * width)]
                selection.SetPixeli(x + (y * int w), c)
        selection
            
    let moveControl (element:FrameworkElement) 
                    (start:Point) (finish:Point) =
        element.Width <- abs(finish.X - start.X)
        element.Height <- abs(finish.Y - start.Y)
        Canvas.SetLeft(element, min start.X finish.X)
        Canvas.SetTop(element, min start.Y finish.Y)

    
    (*  Two state logic
        The initial state is "Waiting", asynchronously awaiting for the event "Mouse Down".
        By pressing the mouse the state will switch to "Drawing". 
        In this state, we can either continue Drawing by moving the mouse, or complete 
        the task and change the state of the application back to "Waiting" by releasing the button.
        The state machine is using asynchronous workflows.
        The function "waiting()" adds a temporary transparent canvas on top of the existing one, 
        with the purpose of providing a visual effect while drawing the rectangle, 
        ultimately, it will represent the Zoom portion.
        The function "drawing" takes as an argument a tuple, with the position "Point" that represents 
        the X and Y coordinates of the corners of the rectangle. 
        Morover, this function is awaiting asynchronously two events
        1)  "Mouse Move" event, which is keeping track of the current position of the mouse 
            while moving and drawing the rectangle
        2) "Mouse Up" event, which is stopping the "drawing" and retrieving the current
            mouse position to use for zooming.
    *)

    

    // TODO:    Build functionality for Zooming out
    let rec waiting() = async {
        let! md = Async.AwaitObservable(self.MouseLeftButtonDown)

        let rc = new Canvas(Background = transparentGray)
        ignore <| canvas.Children.Add(rc) 
        do! drawing(rc, md.GetPosition(canvas)) }

    and drawing(rc:Canvas, pos) = async {
        let! evt = Async.AwaitObservable(canvas.MouseLeftButtonUp, canvas.MouseMove)
        match evt with
        | Choice1Of2(up) ->
            let l, t = Canvas.GetLeft(rc), Canvas.GetTop(rc)
            let w, h = rc.Width, rc.Height
            if w > 1.0 && h > 1.0 then
                let preview = 
                    Image(Source=copy (l,t,w,h),
                          Stretch=Stretch.Fill,
                          Width=float width,Height=float height)
                ignore <| canvas.Children.Add preview

                let zoom (x1,y1,x2,y2) =           
                    let tx x = ((x/float width) * (x2-x1)) + x1
                    let ty y = ((y/float height) * (y2-y1)) + y1
                    tx l, ty t, tx (l+w), ty (t+h)

                points <- zoom points
              
                let pixels = (getPixels(bitmap))

                let sw = System.Diagnostics.Stopwatch.StartNew()

                if runInParallel = false then 
                    do render points (0,height,1) pixels
                else

                // TASK:    Add parallelism functionality
                //          to speed up the rendering


                    let threads = Environment.ProcessorCount
                    do! [0..threads - 1] 
                            |> List.map (fun y ->                         
                                    async { render points (y,(height/threads),threads) pixels
                                })
                            |> Async.Parallel 
                            |> Async.Ignore




                self.Title <- sprintf "Time execution %d ms" sw.ElapsedMilliseconds

                canvas.Children.Remove preview |> ignore

            canvas.Children.Remove rc |> ignore
            do! waiting() 
        | Choice2Of2(move) ->
            moveControl rc pos (move.GetPosition(canvas))
            do! drawing(rc, pos) }
    
    do  waiting() |> Async.StartImmediate

[<STAThread>]
(new Application()).Run(ZoomControl(512,484)) |> ignore  
    