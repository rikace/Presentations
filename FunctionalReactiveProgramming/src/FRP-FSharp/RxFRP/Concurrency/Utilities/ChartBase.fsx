#r "System.Windows.Forms.DataVisualization.dll"

module ChartHelper = 
    open System
    open System.Drawing
    open System.Windows.Forms
    open System.Windows.Forms.DataVisualization.Charting

    let createCPUChart () = 
        let addSeries typ (chart:Chart) =
            let series = new Series(ChartType = typ)
            chart.Series.Add(series)
            series

        let createChart typ =
            let chart = new Chart(Dock = DockStyle.Fill, 
                                  Palette = ChartColorPalette.Pastel)
            let mainForm = new Form(Visible = true, Width = 700, Height = 500)
            mainForm.TopMost <- true
            let area = new ChartArea()
            area.AxisX.MajorGrid.LineColor <- Color.LightGray
            area.AxisY.MajorGrid.LineColor <- Color.LightGray
            mainForm.Controls.Add(chart)
            chart.ChartAreas.Add(area)
            chart, addSeries typ chart

        let chart, series = createChart SeriesChartType.SplineArea
        let area = chart.ChartAreas.[0]
        area.BackColor <- Color.Black
        area.AxisX.MajorGrid.LineColor <- Color.DarkGreen
        area.AxisY.MajorGrid.LineColor <- Color.DarkGreen
        chart.BackColor <- Color.Black
        series.Color <- Color.Green
        series