open System.Drawing
open System.Windows.Forms
open System
open Utils

// Create user interface & list of rectangles & setup drawing
let form = new Form(Visible=true)
let rnd = new Random()
let getRandomColor() = 
    let r, g, b = rnd.Next(256), rnd.Next(256), rnd.Next(256)
    r, g, b

// Takes corners of the rectangle as tuples
let drawRectangle((x1, y1), (x2, y2)) =
   use gr = form.CreateGraphics()
   let r, g, b = getRandomColor()
   use br = new SolidBrush(Color.FromArgb(r,g,b))
   // Calculate upper-left point and rectangle size
   let left, top = min x1 x2, min y1 y2
   let width, height = abs(x1 - x2), abs(y1 - y2)
   gr.Clear(Color.White)
   gr.FillRectangle(br, Rectangle(left, top, width, height))


let rec drawingLoop(point) = async {
   // Wait for the first MouseMove occurrence
   let! info = Async.AwaitObservable(form.MouseMove, form.MouseUp)
   match info with
   | Choice1Of2 move -> 
      // Refresh the window & continue looping
      drawRectangle(point, (move.X, move.Y))
      return! drawingLoop(point)
   | Choice2Of2 up ->
      // Return the end position of rectangle
      return! waitingLoop() }

and waitingLoop() = async {
   while true do
      let! down = Async.AwaitObservable(form.MouseDown)
      let downPointPos = (down.X, down.Y)
      return! drawingLoop(downPointPos) }


[<STAThread>]
do Async.StartImmediate(waitingLoop()) // Starting state 'waitingLoop'
   Application.Run(form)