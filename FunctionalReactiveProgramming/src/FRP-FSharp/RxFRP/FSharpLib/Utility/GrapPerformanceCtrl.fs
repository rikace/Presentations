namespace Easj360FSharp

module GrapPerformanceCtrl =

    open System
    open System.Drawing
    open System.Drawing.Drawing2D
    open System.Windows.Forms
    open System.ComponentModel

    type Sample = { Time : int64; Value : float32 }

    type DataSamples() =
        let data = new ResizeArray<Sample>()
        let mutable count = 0
        let mutable lastTime = 0L

        member x.Last = { Time=lastTime; Value=data.[data.Count - 1].Value }

        member x.AddSample(t,v) =
            let s = { Time=t; Value=v }
            let last = if (data.Count = 0) then s else x.Last

            count <- count + 1
            lastTime <- max last.Time s.Time
            if data.Count = 0 then data.Add(s)

            elif last.Time < s.Time && last.Value <> s.Value then
                if data.[data.Count-1].Time <> last.Time then data.Add(last)
                data.Add(s)

        member x.Count = count

        // The model is continuous: missing samples are obtained by interpolation
        member x.GetValue(time:int64) =

            // Find the relevant point via a binary search
            let rec search (lo, hi) =
                let mid = (lo + hi) / 2
                if hi - lo <= 1 then (lo, hi)
                elif data.[mid].Time = time then (mid, mid)
                elif data.[mid].Time < time then search (mid, hi)
                else search (lo, mid)

            if (data.Count = 0) then failwith "No data samples"

            if (lastTime < time) then failwith "Wrong time!"

            let lo,hi = search (0, data.Count - 1)

            if (data.[lo].Time = time || hi = lo) then data.[lo].Value
            elif (data.[hi].Time = time) then data.[hi].Value
            else
                // interpolate
                let p = if data.[hi].Time < time then hi else lo
                let next = data.[min (p+1) (data.Count-1)]
                let curr = data.[p]
                let spant = next.Time - curr.Time
                let spanv = next.Value - curr.Value
                curr.Value + float32(time-curr.Time) *(spanv/float32(spant))

        // This method finds the minimum and the maximum values given
        // a sampling frequencye and an interval of time
        member x.FindMinMax(sampleFreq:int64, start:int64, finish:int64,
                            minval:float32, maxval:float32) =


            if (data.Count = 0) then (minval, maxval) else
            let start = max start 0L
            let finish = min finish lastTime

            let minv,maxv =
                { start .. sampleFreq .. finish }
                |> Seq.map x.GetValue
                |> Seq.fold (fun (minv,maxv) v -> (min v minv,max v maxv))
                            (minval,maxval)

            if (minv = maxv) then
                let delta = if (minv = 0.0f) then 0.01f else 0.01f * abs minv
                (minv - delta, maxv + delta)
            else (minv, maxv)

    type GraphControl() as x  =
        inherit UserControl()

        let data = new DataSamples()
        let mutable minVisibleValue = Single.MaxValue
        let mutable maxVisibleValue = Single.MinValue
        let mutable absMax          = Single.MinValue
        let mutable absMin          = Single.MaxValue
        let mutable lastMin         = minVisibleValue
        let mutable lastMax         = maxVisibleValue
        let mutable axisColor       = Color.White
        let mutable beginColor      = Color.Red
        let mutable verticalLabelFormat = "{0:F2}"
        let mutable startTime       = 0L
        let mutable visibleSamples  = 10
        let mutable initView = startTime - int64(visibleSamples)
        let mutable verticalLines     = 0
        let mutable timeScale = 10000000 // In 100-nanoseconds
        let mutable timeFormat = "{0:T}"

        let rightBottomMargin = Size(10, 10)
        let leftTopMargin     = Size(10, 10)

        do
          x.SetStyle(ControlStyles.AllPaintingInWmPaint, true)
          x.SetStyle(ControlStyles.OptimizedDoubleBuffer, true)
          base.BackColor <- Color.DarkBlue

        [<Category("Graph Style"); Browsable(true)>]
        member x.AxisColor
            with get() = axisColor
            and set(v:Color) = axisColor <- v; x.Invalidate()

        [<Category("Graph Style"); Browsable(true)>]
        member x.BeginColor
            with get() = beginColor
            and set(v:Color) = beginColor <- v; x.Invalidate()

        [<Category("Graph Style"); Browsable(true)>]
        member x.MinVisibleValue
            with get() = minVisibleValue
            and set(v:float32) = minVisibleValue <- v; lastMin <- v; x.Invalidate()

        [<Category("Graph Style"); Browsable(true)>]
        member x.MaxVisibleValue
            with get() = maxVisibleValue
            and set(v:float32) = maxVisibleValue <- v; lastMax <- v; x.Invalidate()

        [<Category("Graph Style"); Browsable(true)>]
        member x.VerticalLines
            with get() = verticalLines
            and set(v:int) = verticalLines <- v; x.Invalidate()


        [<Category("Graph Style"); Browsable(true)>]
        member x.GraphBackColor
            with get() = x.BackColor
            and set(v:Color) = x.BackColor <- v

        [<Category("Graph Style"); Browsable(true)>]
        member x.LineColor
            with get() = x.ForeColor
            and set(v:Color) = x.ForeColor <- v

        [<Category("Graph Style"); Browsable(true)>]
        member x.VerticalLabelFormat
            with get() = verticalLabelFormat
            and set(v:string) = verticalLabelFormat <- v; x.Invalidate()

        [<Category("Graph Style"); Browsable(true)>]
        member x.StartTime
            with get() = startTime
            and set(v:int64) = startTime <- v; x.Invalidate()

        [<Category("Graph Style"); Browsable(true)>]
        member x.Title
            with get() = x.Text
            and set(v:string) = x.Text <- v; x.Invalidate()

        [<Category("Graph Style"); Browsable(true)>]
        member x.VisibleSamples
            with get() = visibleSamples
            and set(v:int) =
                visibleSamples <- v;
                initView <- startTime - int64(visibleSamples);
                x.Invalidate()

        [<Category("Graph Style"); Browsable(true)>]
        member x.TimeScale
            with get() = timeScale
            and set(v:int) = timeScale <- v; x.Invalidate()

        [<Category("Graph Style"); Browsable(true)>]
        member x.TimeFormat
            with get() = timeFormat
            and set(v:string) = timeFormat <- v; x.Invalidate()

        override x.OnMouseWheel (e:MouseEventArgs) =
            base.OnMouseWheel(e)
            x.Zoom(e.Delta)

        override x.OnSizeChanged (e:EventArgs) =
            base.OnSizeChanged(e)
            x.Invalidate()

        member x.Zoom (amount:int) =
            let newVisibleSamples = max 5 (visibleSamples + amount)
            if (initView - startTime < 0L) then
                let e = initView + int64(visibleSamples)
                initView <- startTime - int64(newVisibleSamples) + e
                visibleSamples <- newVisibleSamples
                x.Invalidate()

        member x.AddSample (time:int64, value:float32) =
            if (value < absMin) then absMin <- value
            if (value > absMax) then absMax <- value
            if (data.Count > 0) then
                initView <- initView + time - data.Last.Time
            data.AddSample(time, value)
            x.Invalidate()

        member x.GetTime (time:int64) =
            DateTime(max 0L time * int64(timeScale))

        override x.OnPaint (e:PaintEventArgs) =
            let g = e.Graphics

            // A helper function to size up strings
            let measurestring s = g.MeasureString(s, x.Font)

            // Work out the size of the box to show the values
            let valBox =
                let minbox = measurestring (String.Format(verticalLabelFormat, lastMin))
                let maxbox = measurestring (String.Format(verticalLabelFormat, lastMax))
                let vbw = max minbox.Width maxbox.Width
                let vbh = max minbox.Height maxbox.Height
                SizeF(vbw, vbh)

            // Work out the size of the box to show the times
            let timeBox =
                let lasttime = x.GetTime(initView + int64(visibleSamples))
                let timelbl = String.Format(timeFormat, lasttime)
                measurestring timelbl

            // Work out the plot area for the graph
            let plotBox =
                let ltm = leftTopMargin
                let rbm = rightBottomMargin

                let ltm,rbm =
                    let ltm = Size(width=max ltm.Width (int(valBox.Width)+5),
                                   height=max ltm.Height (int(valBox.Height/2.0f) + 2))
                    let rbm = Size(width=rightBottomMargin.Width,
                                   height=max rbm.Height (int(timeBox.Height) + 5))
                    ltm,rbm

                // Since we invert y axis use Top instead of Bottom and vice versa
                Rectangle(ltm.Width, rbm.Height,
                          x.Width - ltm.Width - rbm.Width,
                          x.Height - ltm.Height - rbm.Height)

            // The time interval per visible sample
            let timePerUnit =
                let samplew = float32(visibleSamples) / float32(plotBox.Width)
                max 1.0f samplew

            // The pixel interval per visible sample
            let pixelsPerUnit =
                let pixelspan = float32(plotBox.Width) / float32(visibleSamples)
                max 1.0f pixelspan

            // Compute the range we need to plot
            let (lo, hi) = data.FindMinMax(int64(timePerUnit),
                                           initView,
                                           initView + int64(visibleSamples),
                                           minVisibleValue,
                                           maxVisibleValue)

            // Save the range to help with computing sizes next time around
            lastMin <- lo; lastMax <- hi

            // We use these graphical resources during plotting
            use linePen  = new Pen(x.ForeColor)
            use axisPen  = new Pen(axisColor)
            use beginPen = new Pen(beginColor)
            use gridPen  = new Pen(Color.FromArgb(127, axisColor),
                                   DashStyle=DashStyle.Dash)
            use fontColor = new SolidBrush(axisColor)

            // Draw the title
            if (x.Text <> null && x.Text <> String.Empty) then

                let sz = measurestring x.Text
                let mw = (float32(plotBox.Width) - sz.Width) / 2.0f
                let tm = float32(plotBox.Bottom - plotBox.Height)

                let p = PointF(float32(plotBox.Left) + mw, tm)
                g.DrawString(x.Text, x.Font, new SolidBrush(x.ForeColor), p)

            // Draw the labels
            let nly = int((float32(plotBox.Height) /valBox.Height) / 3.0f)
            let nlx = int((float32(plotBox.Width) / timeBox.Width) / 3.0f)
            let pxly = plotBox.Height / max nly 1
            let pxlx = plotBox.Width / max nlx 1
            let dvy = (hi - lo) / float32(nly)
            let dvx = float32(visibleSamples) / float32(nlx)
            let drawString (s:string) (xp:float32) (yp:float32) =
                g.DrawString(s,x.Font,fontColor,xp,yp)

            // Draw the value (y) labels
            for i = 0 to nly do
              let liney = i * pxly + int(valBox.Height / 2.0f) + 2
              let lblfmt = verticalLabelFormat
              let posy = float32(x.Height - plotBox.Top - i * pxly)
              let label = String.Format(lblfmt, float32(i) * dvy + lo)
              drawString label (float32(plotBox.Left) - valBox.Width)
                               (posy - valBox.Height / 2.0f)

              if (i = 0 ||((i > 0) && (i < nly))) then
                g.DrawLine(gridPen, plotBox.Left,liney,plotBox.Right, liney)

            // Draw the time (x) labels
            for i = 0 to nlx do
              let linex = i * pxlx + int(timeBox.Width / 2.0f) + 2
              let time = int64(float32(i) * dvx + float32(initView))
              let label = String.Format(timeFormat, x.GetTime(time))

              if (time > 0L) then
                drawString label
                     (float32(plotBox.Left + i * pxlx) + timeBox.Width / 2.0f)
                     (float32(x.Height - plotBox.Top + 2))

            // Set a transform on the graphics state to make drawing in the
            // plotBox simpler
            g.TranslateTransform(float32(plotBox.Left),
                                  float32(x.Height - plotBox.Top));
            g.ScaleTransform(1.0f, -1.0f);

            // Draw the plotBox of the plot area
            g.DrawLine(axisPen, 0, 0, 0, plotBox.Height)
            g.DrawLine(axisPen, 0, 0, plotBox.Width, 0)
            g.DrawLine(axisPen, plotBox.Width, 0, plotBox.Width, plotBox.Height)
            g.DrawLine(axisPen, 0, plotBox.Height, plotBox.Width, plotBox.Height)


            // Draw the vertical lines in the plotBox
            let px = plotBox.Width / (verticalLines + 1)
            for i = 1 to verticalLines do
                g.DrawLine(gridPen, i*px, 0, i*px, plotBox.Height)

            // Draw the 'begin' marker that shows where data begins
            if (initView - startTime <= 0L) then
                let off = float32(Math.Abs(x.StartTime - initView))
                let sx = int((off/timePerUnit) * pixelsPerUnit)
                g.DrawLine(beginPen, sx, 0, sx, plotBox.Height)

            // Draw the 'zero' horizontal line if it's visible
            if (hi <> lo && lo < 0.0f) then
                let sy = int((float32(plotBox.Height)/(hi - lo))*(0.0f - lo))
                g.DrawLine(axisPen, 0, sy, plotBox.Width, sy)

            // Draw the visible data samples
            let rec drawSamples i pos =
                if (i < (float32(plotBox.Width) / pixelsPerUnit) &&
                    pos <= (initView + int64 visibleSamples - int64 timePerUnit)) then

                    if (pos >= 0L) then
                        let dh = float32(plotBox.Height) / (hi - lo)
                        let sx = int(pixelsPerUnit * i)
                        let dx = int(pixelsPerUnit * (i + 1.0f))
                        let sy = int(dh * (data.GetValue(pos) - lo))
                        let dy = int(dh * (data.GetValue(pos + int64 timePerUnit) - lo))
                        g.DrawLine(linePen, sx, sy, dx, dy);

                    drawSamples (i + 1.0f) (pos + int64 timePerUnit)

            drawSamples 0.0f initView

    let form = new Form(Text="Chart test",Size=Size(800, 600),Visible=true,TopMost=true)
    let graph = new GraphControl(VisibleSamples=60, Dock=DockStyle.Fill)
    let properties = new PropertyGrid(Dock=DockStyle.Fill)
    let timer = new Timer(Interval=200)
    let container = new SplitContainer(Dock=DockStyle.Fill, SplitterDistance=350)

    // We use a split container to divide the area into two parts
    container.Panel1.Controls.Add(graph)
    container.Panel2.Controls.Add(properties)


    // Configure the property grid to display only properties in the
    // category "Graph Style"
    properties.SelectedObject <- graph
    let graphStyleCat = (CategoryAttribute("Graph Style") :> Attribute)
    properties.BrowsableAttributes <- AttributeCollection([| graphStyleCat |])
    form.Controls.Add(container)
    let rnd = new Random()
    let time = ref 0
    // A timer is used to simulate incoming data
    timer.Tick.Add(fun _ ->
        incr time
        let v = 48.0 + 2.0 * rnd.NextDouble()
        graph.AddSample(int64(!time),float32(v)))
    timer.Start()
    form.Disposed.Add(fun _ -> timer.Stop())

    [<STAThread>]
    do Application.Run(form)
