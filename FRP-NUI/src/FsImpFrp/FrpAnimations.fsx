#r "bin/Debug/FRPFSharp.dll"
#r "System.Windows.Forms"
#r "System.Drawing"

open FRPFSharp
open Primitives
open System.Drawing
open System.Drawing.Imaging 
open System.Windows.Forms
open System

type Drawing =
  abstract Draw : Graphics -> unit
  
let drawing f =
      { new Drawing with 
          member x.Draw(gr) = f(gr) }
  

// Creating circle 
module Drawings = 
  let circle brush size =
    // Create drawing using higher order function
    drawing(fun g ->   
      g.FillEllipse(brush, -size/2.0f, -size/2.0f, size, size))

  // Return a new translated drawing 
  let translate x y (img:Drawing) =
    drawing (fun g -> 
      g.TranslateTransform(x, y)
      img.Draw(g)
      g.TranslateTransform(-x, -y) )

  let compose (img1:Drawing) (img2:Drawing) = 
    drawing(fun g ->
      img1.Draw(g)
      img2.Draw(g) )



let circle (br:Behavior<Brush>) (size:Behavior<float32>) = 
  Behavior.lift2 Drawings.circle br size

// Custom operator for composing animations
let (.|.) (img1:Behavior<Drawing>) img2 = 
  Behavior.lift2 Drawings.compose img1 img2

// Lifted version of the translate primitive for drawings  
let translate x y img = 
  Behavior.lift3 Drawings.translate x y img




type AnimationForm() as x =
  inherit Form()
  let emptyAnim = constant(drawing(fun _ -> ()))
  let mutable startTime = DateTime.Now
  let mutable anim = emptyAnim
  let mutable shadows = true
  
  do 
    x.SetStyle(ControlStyles.AllPaintingInWmPaint ||| ControlStyles.OptimizedDoubleBuffer, true)
    let tmr = new Timers.Timer(Interval = 25.0)
    tmr.Elapsed.Add(fun _ -> x.Invalidate() )
    tmr.Start()
  
  member x.Animation 
    with get() = anim
    and set(newAnim) = 
      anim <- newAnim
      startTime <- DateTime.Now

  member x.Shadows with get() = shadows and set(v) = shadows <- v
  
  // Redraw the form
  override x.OnPaint(e) =
    e.Graphics.FillRectangle(Brushes.White, Rectangle(Point(0,0), x.ClientSize))
    
    let start = if shadows then 5 else 0
    for t = start downto 0 do
        // Calculate time that we want to draw
        let time = float32 (DateTime.Now - startTime).TotalSeconds
        // Take earlier time when drawing shadows
        let time = time - (float32 t) / 4.0f
        
        // Create a bitmap & draw the animation
        let bmp = new Bitmap(x.ClientSize.Width, x.ClientSize.Height)
        let gr = Graphics.FromImage(bmp)
        gr.TranslateTransform(float32 x.ClientSize.Width/2.f, float32 x.ClientSize.Height/2.f)
        let drawing = readValue(anim, time) 
        drawing.Draw(gr)     
        
        // Use transformation to add alpha-blending of shadows
        let op = if t = 0 then 1.0f else 0.6f - (float32 t) / 10.0f
        let ar = 
          [| 
            [|1.0f; 0.0f; 0.0f; 0.0f; 0.0f |]
            [|0.0f; 1.0f; 0.0f; 0.0f; 0.0f |]
            [|0.0f; 0.0f; 1.0f; 0.0f; 0.0f |]
            [|0.0f; 0.0f; 0.0f; op; 0.0f |]
            [|0.0f; 0.0f; 0.0f; 0.0f; 1.0f |]
          |]
        let clrMatrix = new ColorMatrix(ar);
        let imgAttributes = new ImageAttributes();
        imgAttributes.SetColorMatrix(clrMatrix, ColorMatrixFlag.Default, ColorAdjustType.Bitmap);    
        gr.Dispose()
                
        e.Graphics.DrawImage
          (bmp, Rectangle(Point(0,0), x.ClientSize), 0, 0, bmp.Width, 
           bmp.Height, GraphicsUnit.Pixel, imgAttributes)
    




// -------------------------------------------------------------------------

    // Behavior representing a square of the current time
    let squared = time |> Behavior.map (fun n -> n * n)
    // Get the value after 9 seconds
    readValue(squared, 9.0f) |> ignore



module DrawingSamples =
  open Drawings

  // Create a green and blue circle
  let circleOne = circle Brushes.OrangeRed 100.0f
  let circleTwo = circle Brushes.Purple 100.0f

  // Compose two translated circles
  let drawingCircles = 
     compose (translate -35.0f 35.0f circleOne)
             (translate 35.0f -35.0f circleTwo)

  // Create & display animation
  let animDrawing = constant drawingCircles
  
  let af = new AnimationForm(ClientSize = Size(600, 600), Visible=true)
  af.Animation <- animDrawing
  

  // Lift the 'translate' function to work with behaviors
  let translate x y img = Behavior.lift3 Drawings.translate x y img

  // Multiply 'circularAnim' to get a value in range -100 .. 100
  let circular100 = Behavior.lift2 (*) circularAnim (100.0f |> constant)

  
  // Create and display the animation
  af.Animation <- translate circular100 (constant 0.0f) animDrawing       
    
  


module SolarSystem =
    // Implementing rotation
    let rotate (dist:float32) speed img = 
      // Oscillate between -dist and +dist
      let pos = circularAnim * (constant dist)
      // Delay the Y-coordinate animation by one half
      // we can implement the rotate function using 
      // translate
      img |> translate pos (wait 0.5f pos) |> faster speed

    let sun   = circle (constant Brushes.Orange) (100.0f |> constant)
    let earth = circle (constant Brushes.ForestGreen) (50.0f |> constant)
    let mars  = circle (constant Brushes.BlueViolet) (40.0f |> constant)
    let pluto  = circle (constant Brushes.SteelBlue) (50.0f |> constant)
    let moon  = circle (constant Brushes.DimGray) (10.0f |> constant)
 
    let planets = 
       sun .|. (earth .|. (moon |> rotate 40.0f 12.0f) 
               |> rotate 160.0f 1.3f)
           .|. (mars |> rotate 250.0f 0.7f)
          // .|. (pluto |> rotate 285.0f 0.25f)
           
           .|. (pluto .|. ( (List.init 8 (fun i -> moon |> rotate (10.0f * float32 i) (float32 i + 8.0f))) |> List.reduce (.|.) ) |> rotate 285.0f 0.25f)
           

       |> faster 0.2f
   
    let af = new AnimationForm(ClientSize = Size(750, 750), Visible=true)
    af.Animation <- planets


  
module SwitchTest =
     
    let af = new AnimationForm(ClientSize = Size(750, 750), Visible=true)

    let rotate (dist:float32) speed img =
      let pos = circularAnim * (constant dist)
      img |> translate pos (wait 0.5f pos) |> faster speed
  
    // Create rotating circle with a constant size
    let myCircle = circle (constant Brushes.BlueViolet) (100.0f |> constant)
    let rotatingCircle = rotate 100.0f 1.0f myCircle
    let always x = (fun _ -> x)


    // Event carrying behaviors as a value
    let circleEvt = 
          af.Click 
          |> Observable.map (always 0.1f) 
          // Adds 0.1 to the initial speed 0.0 
          |> Observable.scan (+) 0.0f 
          // Create a new faster animation
          |> Observable.map (fun x -> faster x rotatingCircle)
  
    // Initial animation is suspended
    let init = faster 0.0f rotatingCircle
    // Animation that speeds-up with clicks
    af.Animation <- switch init circleEvt





