#nowarn "25"
#r "..\..\Lib\Microsoft.Accelerator.dll"
#load "..\CommonModule.fsx"
//#load "..\Utilities\show.fs"
#r "FSharp.PowerPack.dll"


open System
open Microsoft.FSharp
open Microsoft.FSharp.Math
open Microsoft.ParallelArrays
open System.Drawing
open System.Windows.Forms
open Common
 
let SimpleTest =
    let curveArray = Array2D.map float32 (array2D[[0.05; 6.0]; [0.052; 12.0];[0.053; 18.0];[0.055; 24.0];[0.054; 30.0];[0.050; 36.0]; [0.053; 42.0];[0.053; 48.0]; [0.052; 54.0];[0.051; 60.0]])
    let shockArray = Array2D.map float32 (array2D [ [ 1.01; 0.0 ]; [ 0.0; 1.0 ] ])
 
    use inputCurve = new FloatParallelArray(curveArray)
    use shockCurve = new FloatParallelArray(shockArray)
 
    let sum = ParallelArrays.InnerProduct(inputCurve, shockCurve) 
 
    use dx9Target = new DX9Target()
    let res = dx9Target.ToArray2D(sum)
    printfn "result %A" sum
    


module GameOfLife =
 
    // Type aliases for the most frequently used Accelerator types
    type PA = Microsoft.ParallelArrays.ParallelArrays
    type FPA = Microsoft.ParallelArrays.FloatParallelArray

    // A form that runs a loop - updating a state, drawing a new state...
    type DrawingForm<'TState>(initial, update, draw, handleClick, size, title) as x = 
      inherit Form(ClientSize = Size(size, size), Text = title)
      let mutable clicked = 0
      // Loop that runs computation & redraws form
      let rec drawingLoop(state) = 
        if not x.IsDisposed then
          // Calculate the next state - this also calls the click 
          // handler if the button was clicked since the last time
          let state = 
            if clicked <> 0 then 
              let res = handleClick (clicked = 1) state
              clicked <- 0
              res
            else update(state)
          // Do the drawing & continue looping
          ( use gr = x.CreateGraphics()
            gr.InterpolationMode <- Drawing2D.InterpolationMode.NearestNeighbor
            gr.DrawImage(draw(state), x.ClientRectangle) )
          drawingLoop(state)

      do
        // Register handlers & run the computation
        x.MouseDown.Add(fun e -> clicked <- if e.Button = MouseButtons.Left then 1 else 2)
        Async.Start(async { drawingLoop(initial) })
 
    //-----------------------------------------------------------------------------
    // LIFE GAME
    //-----------------------------------------------------------------------------

    // Configuration of the game
    let gridSize, formSize = 128, 512

    // Initialization of constant 2D arrays
    let shape = [| gridSize; gridSize; |]
    let [zero; one; two; three] =
      [ for f in [0.0f; 1.0f; 2.0f; 3.0f] -> new FPA(f, shape) ]

    //-----------------------------------------------------------------------------

    // Custom operators - simulating logical operations using floats
    let (&&.) (a:FPA) (b:FPA) = PA.Min(a, b)
    let (||.) (a:FPA) (b:FPA) = PA.Max(a, b)
    let (==.) (a:FPA) (b:FPA) = PA.Cond(PA.CompareEqual(a, b), one, zero)

    // Calculating with 2D float arrays 
    let rotate (a:FPA) (dx:int) dy = PA.Rotate(a, [| dx; dy |]);

    //-----------------------------------------------------------------------------

    // Create an initial data grid
    let initial = Array2D.init gridSize gridSize (fun x y -> 
      if x = 0 || y = 0 || x = y then 1.0f else 0.0f)

    // Calculate FPA representing the next generation 
    let nextGeneration (num:FloatParallelArray) =
      // Generate list with grids shifted by +/-1 in both directions
      let hd::tl = 
        [ for dx in -1 .. 1 do
            for dy in -1 .. 1 do
              if dx <> 0 || dy <> 0 then yield rotate num dx dy ]
      // Sum all grids to get count of neighbors
      let sum = tl |> List.fold (+) hd
      // Keep alive if number of neighbors is 2 or 3
      (sum ==. three) ||. ((sum ==. two) &&. num)


    /// Runs the game of life using the specified computation engine
    let life(target:Target, title) =

      // Calculate the next state as 2D array of floats
      let nextState (g:float32[,]) =
        let fp = new FloatParallelArray(g)
        target.ToArray2D(nextGeneration(fp))
  
      // Draw 2D float array as a bitmap
      let display (data:float32[,]) =
        data |> toBitmap (fun f -> 
          if (f < 0.5f) then Color.White else Color.DarkCyan)

      // If the user clicks with left button, add some new organisms;
      // for right button, kill some of them (with probability 1/20)
      let clickHandler left (g:float32[,]) =
        let rnd = new Random()
        g |> Array2D.map (fun orig ->
          match rnd.Next(20), left with
          | 0, true -> 1.0f
          | 0, false -> 0.0f
          | _ -> orig )
    
      // Run the drawing form with life game
      let b = new DrawingForm<_>(initial, nextState, display, clickHandler, formSize, title)
      b.ShowDialog() |> ignore
  
//-----------------------------------------------------------------------------
  
// Create multicore engine using x64 multicore
let multiCoreTarget = new MulticoreTarget() 
GameOfLife.life(multiCoreTarget, "Multi Cores")  

// Create computation engine using DirectX 9
let dxTarget = new DX9Target()
GameOfLife.life(dxTarget, "GPU")  
