// ----------------------------------------------------------------------------

// This section implements two helper functions. The function createChart 
// constructs a new chart object with a single chart area and adds a new data 
// series to the area. The addSeries function creates a new series using the 
// specified chart type and adds it to a given chart

#r "System.Windows.Forms.DataVisualization.dll"

open System
open System.Drawing
open System.Windows.Forms
open System.Windows.Forms.DataVisualization.Charting

/// Add data series of the specified chart type to a chart
let addSeries typ (chart:Chart) =
    let series = new Series(ChartType = typ)
    chart.Series.Add(series)
    series

/// Create form with chart and add the first chart series
let createChart typ =
    let chart = new Chart(Dock = DockStyle.Fill, 
                          Palette = ChartColorPalette.Pastel)
    let mainForm = new Form(Visible = true, Width = 700, Height = 500)
    let area = new ChartArea()
    area.AxisX.MajorGrid.LineColor <- Color.LightGray
    area.AxisY.MajorGrid.LineColor <- Color.LightGray
    mainForm.Controls.Add(chart)
    chart.ChartAreas.Add(area)
    chart, addSeries typ chart

// ----------------------------------------------------------------------------
// Showing CPU usage

open System.Diagnostics

module CpuUsage =

  /// Function that returns the current CPU usage
  let getCpuUsage = 
      let counter = 
          new PerformanceCounter
            ( CounterName = "% Processor Time",
              CategoryName = "Processor", InstanceName = "_Total" )
      (fun () -> counter.NextValue())

  // Create objects representing the chart. The tutorial uses a helper function 
  // createChart that is implemented in the previous tutorial. The following 
  // snippet configures the chart to use a spline area chart

  let chart, series = createChart SeriesChartType.SplineArea
  let area = chart.ChartAreas.[0]
  area.BackColor <- Color.Black
  area.AxisX.MajorGrid.LineColor <- Color.DarkGreen
  area.AxisY.MajorGrid.LineColor <- Color.DarkGreen
  chart.BackColor <- Color.Black
  series.Color <- Color.Green

  // The next snippet creates an asynchronous workflow that periodically 
  // updates the chart area. The implementation uses a while loop in the 
  // workflow and the workflow is started such that all user code runs 
  // on the main GUI thread

  let updateLoop = async { 
      while not chart.IsDisposed do
          let v = float (getCpuUsage()) 
          series.Points.Add(v) |> ignore
          do! Async.Sleep(250) }
  Async.StartImmediate updateLoop

// ----------------------------------------------------------------------------
// Retrieving data in the background

open System.Threading

module RandomWalk = 

  let chart, series = createChart SeriesChartType.FastLine
  let axisX = chart.ChartAreas.[0].AxisX
  let axisY = chart.ChartAreas.[0].AxisY
  chart.ChartAreas.[0].InnerPlotPosition <- 
      new ElementPosition(10.0f, 2.0f, 85.0f, 90.0f)

  // The following function takes the number of points generated so far as 
  // the argument. It finds the maximal and minimal Y value in the series 
  // and updates the ranges

  let updateRanges(n) =    
    let values = seq { for p in series.Points -> p.YValues.[0] }
    axisX.Minimum <- float n - 500.0
    axisX.Maximum <- float n 
    axisY.Minimum <- values |> Seq.min |> Math.Floor
    axisY.Maximum <- values |> Seq.max |> Math.Ceiling

  // The following asynchronous function switches to the GUI thread, checks 
  // if the chart is still open, updates the chart, and then switches back 
  // to a background thread. It also returns a Boolean value indicating 
  // whether or not the chart has been disposed, so that the calling 
  // workflow can terminate when the chart is closed

  let ctx = SynchronizationContext.Current

  let updateChart(valueX, valueY) = async {
      do! Async.SwitchToContext(ctx)
      if chart.IsDisposed then 
          do! Async.SwitchToThreadPool()
          return false
      else  
          series.Points.AddXY(valueX, valueY) |> ignore
          while series.Points.Count > 500 do series.Points.RemoveAt(0)
          updateRanges valueX
          do! Async.SwitchToThreadPool()
          return true }

  // The next snippet implements a recursive asynchronous loop that 
  // performs the random walk (with blocking on a background thread)

  let randomWalk =
      let rnd = new Random()
      let rec loop(count, value) = async {
          let count, value = count + 1, value + (rnd.NextDouble() - 0.5)
          Thread.Sleep(20)
          let! running = updateChart(float count, value)
          if running then return! loop(count, value) }
      loop(0, 0.0)

  Async.Start(randomWalk)


module CpuUsageChart = 

  let chart, series = createChart SeriesChartType.FastLine
  let axisX = chart.ChartAreas.[0].AxisX
  let axisY = chart.ChartAreas.[0].AxisY
  axisY.Minimum <- 0.
  axisY.Maximum <- 100.
  chart.ChartAreas.[0].InnerPlotPosition <- 
      new ElementPosition(10.0f, 2.0f, 85.0f, 90.0f)

        /// Function that returns the current CPU usage
  let getCpuUsage = 
      let counter = 
          new PerformanceCounter
            ( CounterName = "% Processor Time",
              CategoryName = "Processor", InstanceName = "_Total" )
      (fun () -> counter.NextValue())

  // The following function takes the number of points generated so far as 
  // the argument. It finds the maximal and minimal Y value in the series 
  // and updates the ranges

  let updateRanges(n) =    
    let values = seq { for p in series.Points -> p.YValues.[0] }
    axisX.Minimum <- float n - 500.0
    axisX.Maximum <- float n 


  // The following asynchronous function switches to the GUI thread, checks 
  // if the chart is still open, updates the chart, and then switches back 
  // to a background thread. It also returns a Boolean value indicating 
  // whether or not the chart has been disposed, so that the calling 
  // workflow can terminate when the chart is closed

  let ctx = SynchronizationContext.Current

  let updateChart(valueX, valueY) = async {
      do! Async.SwitchToContext(ctx)
      if chart.IsDisposed then 
          do! Async.SwitchToThreadPool()
          return false
      else  
          series.Points.AddXY(valueX, valueY) |> ignore
          while series.Points.Count > 500 do series.Points.RemoveAt(0)
          updateRanges valueX
          do! Async.SwitchToThreadPool()
          return true }

  // The next snippet implements a recursive asynchronous loop that 
  // performs the random walk (with blocking on a background thread)

  let randomWalk =
      let rnd = new Random()
      let rec loop(count) = async {
          let value = float( getCpuUsage() )
          Console.WriteLine(value.ToString())
          Thread.Sleep(20)
          let! running = updateChart(float count, value)
          if running then return! loop(count + 1) }
      loop(0)

  Async.Start(randomWalk)
