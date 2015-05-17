#nowarn "25"
#r "..\..\Lib\Microsoft.Accelerator.dll"
//#load "..\Utilities\show.fs"
#r "FSharp.PowerPack.dll"

open System
open Microsoft.ParallelArrays

//-----------------------------------------------------------------------------
// Type aliases for the most frequently used Accelerator types

type PA = Microsoft.ParallelArrays.ParallelArrays
type FPA = Microsoft.ParallelArrays.FloatParallelArray

//-----------------------------------------------------------------------------

// Configuration of the calculation
let gridSize = 4096

// Initialization of constant 2D arrays
let shape = [| gridSize; gridSize; |]
let zero, one = new FPA(0.0f, shape), new FPA(1.0f, shape)

//-----------------------------------------------------------------------------

// Calculating with 2D float arrays 
let select (a:FPA) (b:FPA) (c:FPA) = PA.Select(a,b,c)
let sqrt (a:FPA) = PA.Sqrt(a)
let sum (a:FPA) = PA.Sum(a)

//-----------------------------------------------------------------------------
// CALCULATING PI
//-----------------------------------------------------------------------------

let rnd = new Random()
let generator x y = float32(rnd.NextDouble())

// Initialize large arrays with X and Y coordinates
// (the arrays are 2D, which allows more parallelism,
// because a GPU internally works with 2D textures)
let px = new FPA(Array2D.init gridSize gridSize generator)
let py = new FPA(Array2D.init gridSize gridSize generator)

let calculatePi (target:Target) =
  
  // Create values with center (0,0) and with range (-1,1)
  let pxCentered = (px - 0.5f) * 2.0f
  let pyCentered = (py - 0.5f) * 2.0f

  // Calculate distance of (x,y) from (0,0), ie. sqrt(x^2 + y^2)
  let distances = sqrt ((pxCentered * pxCentered) + (pyCentered * pyCentered))

  // If distance < 1 then return 1 else return 0 for each cell
  let inCircle = select (one - distances) one zero

  // Number of values in circle is the sum of all 1's in the array
  let count = sum(inCircle)
  
  let result = target.ToArray1D(count)

  // Area of unit circle is PI, Area of square from -1 to 1 is 4
  let pi = result.[0] / (float32(gridSize * gridSize) / 4.0f)
  printfn "Approximate PI value: %f" pi

//-----------------------------------------------------------------------------
open System.Drawing
open System.Windows.Forms

/// This function shows a very simple form with an image that demonstrates
/// how the Monte-carlo method for calculating PI works
let visualization() =
  let testCount = 10000
  let bitmapSize = 150.0f
  let formSize = 300

  // Generate random X and Y points
  let generator i = float32(rnd.NextDouble()), float32(rnd.NextDouble())
  let points = Seq.init testCount generator
  
  // Draw the points on a bitmap
  let bmp = new Bitmap(int bitmapSize, int bitmapSize)
  for x, y in points do
    let vx, vy = (x - 0.5f) * 2.0f, (y - 0.5f) * 2.0f
    let inside = Math.Sqrt(float(vx * vx + vy * vy)) < 1.0
    bmp.SetPixel
      ( int(x * bitmapSize), int(y * bitmapSize), 
        if inside then Color.OliveDrab else Color.DarkRed )
  // Draw a circle on the bitmap
  ( use gr = Graphics.FromImage(bmp)
    use pn = new Pen(Color.Black, 2.0f)
    gr.DrawEllipse(pn, Rectangle(0, 0, int bitmapSize, int bitmapSize)) )
  
  // Display the bitmap in a form
  let f = new Form(ClientSize = Size(formSize, formSize))
  f.Paint.Add(fun e -> 
    e.Graphics.InterpolationMode <- Drawing2D.InterpolationMode.NearestNeighbor
    e.Graphics.DrawImage(bmp, Rectangle(0, 0, formSize, formSize)))
  f.ShowDialog() |> ignore
  //Application.Run(f)
  
//-----------------------------------------------------------------------------
do 
  // Create computation engine using DirectX 9
  let dxTarget = new DX9Target()
  // Create multicore engine using x64 multicore
  let multiCoreTarget = new MulticoreTarget()
  
  calculatePi(dxTarget)  
  calculatePi(multiCoreTarget)  
  visualization()
  