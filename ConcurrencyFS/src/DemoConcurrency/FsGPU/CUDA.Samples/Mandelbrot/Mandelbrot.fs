////////////////////////////////////////////////////////////////////////////
// This file is a modified version of mandelbrot sample from Expert F# book.
// You can download original version using following url:
// http://www.expert-fsharp.com/CodeSamples/Chapter11/Example05/main.fs

(* Copyright 2009 FsGPU Project.
 *
 * Contributors to this file:
 * Alex Slesarenko - http://slesarenko.blogspot.com
 *
 * This file is part of FsGPU.  FsGPU is licensed under the 
 * GNU Library General Public License (LGPL). See License.txt for a complete copy of the
 * license.
 *)

#light
#nowarn "62"

namespace FsGPU.Cuda.Samples

module Mandelbrot

open System
open System.Collections
open System.Threading
open System.Drawing
open System.Windows.Forms
open Microsoft.FSharp.Math
open GASS.CUDA
open GASS.CUDA.Engine
open GASS.CUDA.Types
open System.IO
open System.Runtime.InteropServices
open FsGPU.Cuda
open FsGPU.Cuda.Samples
open FsGPU.Cuda.Samples.MandelbrotKernels
open FsGPU.Helpers

let sqrMod (x:complex) = x.r * x.r + x.i * x.i

let rec mandel maxit (z:Complex) (c: Complex) count =
    if (sqrMod(z) < 4.0) && (count < maxit) then
        mandel maxit ((z * z) + c) c (count + 1)
    else count
    
let RGBtoHSV (r, g, b) =
    let (m:float) = min r (min g b)
    let (M:float) = max r (max g b)
    let delta = M - m
    let posh (h:float) = if h < 0.0 then h + 360.0 else h
    let deltaf (f:float) (s:float) = (f - s) / delta
    if M = 0.0 then (-1.0, 0.0, M) else
        let s = (M - m) / M
        if r = M then (posh(60.0 * (deltaf g b)), s, M)
        elif g = M then (posh(60.0 * (2.0 + (deltaf b r))), s, M)
        else (posh(60.0 * (4.0 + (deltaf r g))), s, M)

let HSVtoRGB (h, s, v) =
    if s = 0.0 then (v, v, v) else
    let hs = h / 60.0
    let i = floor (hs)
    let f = hs - i
    let p = v * ( 1.0 - s )
    let q = v * ( 1.0 - s * f )
    let t = v * ( 1.0 - s * ( 1.0 - f ))
    match int i with
        | 0 -> (v, t, p)
        | 1 -> (q, v, p)
        | 2 -> (p, v, t)
        | 3 -> (p, q, v)
        | 4 -> (t, p, v)
        | _ -> (v, p, q)

let makeColor (r, g, b) =
    Color.FromArgb(int32(r * 255.0), int32(g * 255.0), int32(b * 255.0))

let defaultColor i = makeColor(HSVtoRGB(360.0 * (float i / 250.0), 1.0, 1.0))

let coloring =
    [| defaultColor;
       (fun i -> Color.FromArgb(i, i, i));
       (fun i -> Color.FromArgb(i, 0, 0));
       (fun i -> Color.FromArgb(0, i, 0));
       (fun i -> Color.FromArgb(0, 0, i));
       (fun i -> Color.FromArgb(i, i, 0));
       (fun i -> Color.FromArgb(i, 250 - i, 0));
       (fun i -> Color.FromArgb(250 - i, i, i));
       (fun i -> if i % 2 = 0 then Color.White else Color.Black);
       (fun i -> Color.FromArgb(250 - i, 250 - i, 250 - i))
    |]

let createPalette c =
    Array.init 253 (function
        | 250 -> Color.Black
        | 251 -> Color.White
        | 252 -> Color.LightGray
        | i -> c i)

let mutable palette = createPalette coloring.[0]

let pickColor maxit it =
    palette.[int(250.0 * float it / float maxit)]

type CanvasForm() as x =
    inherit Form()
    do x.SetStyle(ControlStyles.OptimizedDoubleBuffer, true)

    override x.OnPaintBackground(args) = ()

// Creates the Form
let form = new CanvasForm(Width=800, Height=600,Text="Mandelbrot set")

let mutable worker = Thread.CurrentThread

let mutable bitmap = new Bitmap(form.Width, form.Height)
let mutable bmpw = form.Width
let mutable bmph = form.Height

let mutable startsel = Point.Empty
let mutable rect = Rectangle.Empty

let mutable tl = (-3.0, 2.0)
let mutable br = (2.0, -2.0)

let mutable menuIterations = 50

let iterations (tlx, tly) (brx, bry) =
    menuIterations

let timer = new Timer(Interval=100)
timer.Tick.Add(fun _ -> form.Invalidate() )

let run filler (form:#Form) (bitmap:Bitmap) (tlx, tly) (brx, bry) =
    let dx = (brx - tlx) / float bmpw
    let dy = (tly - bry) / float bmph
    let maxit = iterations (tlx, tly) (brx, bry)
    let x = 0
    let y = 0
    let transform x y =
        Complex.Create (tlx + (float x) * dx, tly - (float y) * dy )
    form.Invoke(new MethodInvoker(fun () ->
        form.Text <- sprintf "Mandelbrot set [it: %d] (%f, %f) -> (%f, %f)"
                     maxit tlx tly brx bry
    )) |> ignore
    filler maxit transform
    timer.Enabled <- false

let linearFill (bw:int) (bh:int) maxit map =
        for y = 0 to bh - 1 do
            for x = 0 to bw - 1 do
                let c = mandel maxit Complex.Zero (map x y) 0
                lock bitmap (fun () -> bitmap.SetPixel(x, y, pickColor maxit c))

let rand = new Random(10)

let calcColors (bw:int) (bh:int) maxit map =
    let tl : Complex = map 0 0       // top left conner
    let bl = map 0 (bh-1)            // bottom left conner
    let xOff, yOff = float32(tl.r), float32(tl.i)
    let scale = (float32(tl.i) - float32(bl.i)) / float32(bh)
    let num = bw * bh
    CalcMandelbrot(bw, bh, maxit, xOff, yOff, scale)

let linearFillGPU (bw:int) (bh:int) maxit map =
    let colors = calcColors bw bh maxit map
    let rect = new Rectangle(0, 0, bw, bh);
    lock bitmap (fun () -> 
        let mutable bmpData = null
        try
            bmpData <- bitmap.LockBits(rect, System.Drawing.Imaging.ImageLockMode.ReadWrite,  Imaging.PixelFormat.Format24bppRgb);
            let ptr = bmpData.Scan0
            let bytes  = bmpData.Stride * bh
            let rgbValues = Array.create bytes 0uy
            Marshal.Copy(ptr, rgbValues, 0, bytes)
            
            for y = 0 to bh - 1 do
                for x = 0 to bw - 1 do
                    let index = (y * bw + x)
                    let offset = index * 3
                    let c = colors.[index]
                    //bitmap.SetPixel(x, y, pickColor maxit c))
                    let color = pickColor maxit c
                    rgbValues.[offset] <- color.B
                    rgbValues.[offset + 1] <- color.G
                    rgbValues.[offset + 2] <- color.R
            
            Marshal.Copy(rgbValues, 0, ptr, bytes)
        finally
            bitmap.UnlockBits(bmpData)
        form.Invalidate()
        )        

let blockFill (bw:int) (bh:int) maxit map =
    let rec fillBlock first sz x y =
        if x < bw then
            let c = mandel maxit Complex.Zero (map x y) 0
            lock bitmap (fun () ->
                let g = Graphics.FromImage(bitmap)
                g.FillRectangle(new SolidBrush(pickColor maxit c), x, y, sz, sz);
                g.Dispose()
            )
            fillBlock first sz (if first || ((y / sz) % 2 = 1) then x + sz
                                else x + 2 * sz) y
        elif y < bh then
            fillBlock first sz (if first || ((y / sz) % 2 = 0) then 0 else sz)
                                (y + sz)
        elif sz > 1 then
            fillBlock false (sz / 2) (sz / 2) 0
        
    fillBlock true 64 0 0

let mutable fillFun = linearFillGPU

let clearOffScreen (b : Bitmap) =
    lock b (fun () ->
        use g = Graphics.FromImage(b)
        g.Clear(Color.White))

let paint (g: Graphics) =
    lock bitmap (fun () -> g.DrawImage(bitmap, 0, 0))
    g.DrawRectangle(Pens.Black, rect)
    g.FillRectangle(new SolidBrush(Color.FromArgb(128, Color.White)), rect)

let stopWorker () =
    if worker <> Thread.CurrentThread then
        worker.Abort();
        //worker.Join(2000) |> ignore;
    worker <- Thread.CurrentThread

let drawMandel () =
    let bf = fillFun bmpw bmph
    stopWorker()
    timer.Enabled <- true
    worker <- new Thread(fun () -> run bf form bitmap tl br)
    worker.IsBackground <- true
    worker.Priority <- ThreadPriority.Lowest
    worker.Start()
    
let setCoord (tlx:float, tly:float) (brx:float, bry:float) =
    let ratio = (float bmpw / float bmph)
    let dx = (brx - tlx) / float bmpw
    let dy = (tly - bry) / float bmph
    let mapx x = tlx + float x * dx
    let mapy y = tly - float y * dy
    if ratio * float rect.Height > float rect.Width then
        let nw = int (ratio * float rect.Height )
        rect.X <- rect.X - (nw - rect.Width) / 2;
        rect.Width <- nw
    else
        let nh = int (float rect.Width / ratio)
        rect.Y <- rect.Y - (nh - rect.Height) / 2;
        rect.Height <- nh;
    tl <- (mapx rect.Left, mapy rect.Top);
    br <- (mapx rect.Right, mapy rect.Bottom)

let updateView () =
    setCoord tl br
    rect <- Rectangle.Empty
    stopWorker()
    clearOffScreen bitmap
    drawMandel()

let click (arg:MouseEventArgs) =
    if rect.Contains(arg.Location) then
        updateView()
    else
        form.Invalidate();
        rect <- Rectangle.Empty;
        startsel <- arg.Location

let mouseMove (arg:MouseEventArgs) =
    if arg.Button = MouseButtons.Left then
        let tlx = min startsel.X arg.X
        let tly = min startsel.Y arg.Y
        let brx = max startsel.X arg.X
        let bry = max startsel.Y arg.Y
        rect <- new Rectangle(tlx, tly, brx - tlx, bry - tly)
        form.Invalidate()

let resize () =
    if bmpw <> form.ClientSize.Width ||
       bmph <> form.ClientSize.Height then
        stopWorker()
        rect <- form.ClientRectangle
        bitmap <- new Bitmap(form.ClientSize.Width, form.ClientSize.Height)
        bmpw <- form.ClientSize.Width
        bmph <- form.ClientSize.Height
        updateView()

let zoom amount =
    let r = form.ClientRectangle
    let nw = int(floor(float r.Width * amount))
    let nh = int(floor(float r.Height * amount))
    rect <- Rectangle(r.X - ((nw - r.Width)/2), r.Y - ((nh-r.Height)/2), nw, nh)
    updateView()

type Direction = Top | Left | Right | Bottom

let move (d:Direction) =
    let r = form.ClientRectangle
    match d with
    | Top -> rect <- Rectangle(r.X, r.Y - (r.Height / 10), r.Width, r.Height)
             updateView()
    | Left -> rect <- Rectangle(r.X - (r.Width / 10), r.Y, r.Width, r.Height)
              updateView()
    | Bottom -> rect <- Rectangle(r.X, r.Y + (r.Height / 10), r.Width, r.Height)
                updateView()
    | Right -> rect <- Rectangle(r.X + (r.Width / 10), r.Y, r.Width, r.Height)
               updateView()

let selectDropDownItem (l:ToolStripMenuItem) (o:ToolStripMenuItem) =
    l.DropDownItems |> Seq.cast
    |> Seq.iter (fun (el : ToolStripMenuItem) -> el.Checked <- ((o = el)))

let setFillMode (p:ToolStripMenuItem) (m:ToolStripMenuItem) filler _ =
    if (not m.Checked) then
        selectDropDownItem p m
        fillFun <- filler
        drawMandel()

let setupMenu () =
    let m = new MenuStrip()
    let f = new ToolStripMenuItem("&File")
    let c = new ToolStripMenuItem("&Settings")
    let e = new ToolStripMenuItem("&Edit")
    let ext = new ToolStripMenuItem("E&xit")
    let cols = new ToolStripComboBox("ColorScheme")
    let its = new ToolStripComboBox("Iterations")
    let copy = new ToolStripMenuItem("&Copy")
    let zoomin = new ToolStripMenuItem("Zoom &In")
    let zoomout = new ToolStripMenuItem("Zoom &Out")
    let fillMode = new ToolStripMenuItem("Fill mode")
    let fillModeLinear = new ToolStripMenuItem("Line")
    let fillModeBlock = new ToolStripMenuItem("Block")
    let fillModeGPU = new ToolStripMenuItem("GPU")
    
    let itchg = fun _ ->
        menuIterations <- System.Int32.Parse(its.Text)
        stopWorker()
        drawMandel()
        c.HideDropDown()
    ext.Click.Add(fun _ -> form.Dispose()) |> ignore
    
    copy.Click.Add(fun _ -> Clipboard.SetDataObject(bitmap))|> ignore
    copy.ShortcutKeyDisplayString <- "Ctrl+C"
    copy.ShortcutKeys <- Keys.Control ||| Keys.C

    zoomin.Click.Add(fun _ -> zoom 0.9) |> ignore
    zoomin.ShortcutKeyDisplayString <- "Ctrl+T"
    zoomin.ShortcutKeys <- Keys.Control ||| Keys.T
    zoomout.Click.Add(fun _ -> zoom 1.25) |> ignore;
    zoomout.ShortcutKeyDisplayString <- "Ctrl+W";
    zoomout.ShortcutKeys <- Keys.Control ||| Keys.W

    for x in [ f;e;c ] do m.Items.Add(x) |> ignore
    f.DropDownItems.Add(ext) |> ignore
    let tsi x = (x :> ToolStripItem)
    for x in [ tsi cols; tsi its; tsi fillMode] do c.DropDownItems.Add(x) |> ignore
    for x in [ tsi copy; tsi zoomin; tsi zoomout ] do e.DropDownItems.Add(x) |> ignore
    for x in ["HSL Color"; "Gray"; "Red"; "Green"] do cols.Items.Add(x) |> ignore
    fillMode.DropDownItems.Add(fillModeLinear) |> ignore
    fillMode.DropDownItems.Add(fillModeBlock) |> ignore
    fillMode.DropDownItems.Add(fillModeGPU) |> ignore
    cols.SelectedIndex <- 0
    cols.DropDownStyle <- ComboBoxStyle.DropDownList
    cols.SelectedIndexChanged.Add(fun _ ->
        palette <- createPalette coloring.[cols.SelectedIndex]
        stopWorker()
        drawMandel()
        c.HideDropDown()
    )
    its.Text <- string_of_int menuIterations
    its.DropDownStyle <- ComboBoxStyle.DropDown
    for x in [ "150"; "250"; "500"; "1000" ] do its.Items.Add(x) |> ignore
    its.LostFocus.Add(itchg)
    its.SelectedIndexChanged.Add(itchg)
    fillModeGPU.Checked <- true
    fillModeLinear.Click.Add(setFillMode fillMode fillModeLinear linearFill);
    fillModeBlock.Click.Add(setFillMode fillMode fillModeBlock blockFill);
    fillModeGPU.Click.Add(setFillMode fillMode fillModeGPU linearFillGPU);
    m

clearOffScreen bitmap
form.MainMenuStrip <- setupMenu()
form.Controls.Add(form.MainMenuStrip)
form.MainMenuStrip.RenderMode <- ToolStripRenderMode.System
form.Paint.Add(fun arg -> paint arg.Graphics)
form.MouseDown.Add(click)
form.MouseMove.Add(mouseMove)
form.ResizeEnd.Add(fun _ -> resize())
form.Show()
Application.DoEvents()
drawMandel()

let TestMandelbrotGPU() = 
    let width = 800
    let height = 600
    let maxit = 50
    let l, b = fst tl, snd br            // bottom left conner
    let xOff, yOff = float32(fst tl), float32(snd tl)
    let scale = (float32(snd tl) - float32(b)) / float32(height)
    // (width : int, height : int, crunch : int, xOff : float32, yOff: float32, scale : float32)
    CalcMandelbrot(width, height, maxit, xOff, yOff, scale)

let TestMandelbrotCPU() = 
    let width = 800
    let height = 600
    let maxit = 50
    let l, b = fst tl, snd br            // bottom left conner
    let xOff, yOff = tl
    let scale = (float32(snd tl) - float32(b)) / float32(600)
    
    let transform x y =
        Complex.Create (xOff + (float x) * (float (scale)), yOff - (float y) * (float scale) )

    let colors = Array.zeroCreate<int> (width * height)
    for y = 0 to height - 1 do
        for x = 0 to width - 1 do
            let i = width * y + x
            colors.[i] <- mandel maxit Complex.Zero (transform x y) 0
    colors

[<STAThread>]
[<EntryPoint>]
let main(args) =
    RunFunc 20 TestMandelbrotGPU "Stress test mandelbrot calculation on GPU" ()
    RunFunc 20 TestMandelbrotCPU "Stress test mandelbrot calculation on CPU" ()
    Application.Run(form)
    0

