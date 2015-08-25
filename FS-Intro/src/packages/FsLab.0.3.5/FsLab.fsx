#nowarn "211"
#I "."
#I "../Deedle/lib/net40"
#I "../Deedle.1.2.2/lib/net40"
#I "../Deedle.RPlugin/lib/net40"
#I "../Deedle.RPlugin.1.2.2/lib/net40"
#I "../FSharp.Charting/lib/net40"
#I "../FSharp.Charting.0.90.12/lib/net40"
#I "../FSharp.Data/lib/net40"
#I "../FSharp.Data.2.2.3/lib/net40"
#I "../Foogle.Charts/lib/net40"
#I "../Foogle.Charts.0.0.5/lib/net40"
#I "../MathNet.Numerics/lib/net40"
#I "../MathNet.Numerics.3.7.0/lib/net40"
#I "../MathNet.Numerics.FSharp/lib/net40"
#I "../MathNet.Numerics.FSharp.3.7.0/lib/net40"
#I "../RProvider/lib/net40"
#I "../RProvider.1.1.12/lib/net40"
#I "../R.NET.Community/lib/net40"
#I "../R.NET.Community.1.6.4/lib/net40"
#I "../R.NET.Community.FSharp/lib/net40"
#I "../R.NET.Community.FSharp.1.6.4/lib/net40"
#I "../XPlot.Plotly/lib/net45"
#I "../XPlot.Plotly.1.2.0/lib/net45"
#I "../XPlot.GoogleCharts/lib/net45"
#I "../XPlot.GoogleCharts.1.2.0/lib/net45"
#I "../XPlot.GoogleCharts.Deedle/lib/net45"
#I "../XPlot.GoogleCharts.Deedle.1.2.0/lib/net45"
#I "../Google.DataTable.Net.Wrapper/lib"
#I "../Google.DataTable.Net.Wrapper.3.1.2.0/lib"
#I "../Newtonsoft.Json/lib/net40"
#I "../Newtonsoft.Json.7.0.1/lib/net40"
#r "Deedle.dll"
#r "Deedle.RProvider.Plugin.dll"
#r "System.Windows.Forms.DataVisualization.dll"
#r "FSharp.Charting.dll"
#r "FSharp.Data.dll"
#r "Foogle.Charts.dll"
#r "MathNet.Numerics.dll"
#r "MathNet.Numerics.FSharp.dll"
#r "RProvider.Runtime.dll"
#r "RProvider.dll"
#r "RDotNet.dll"
#r "RDotNet.NativeLibrary.dll"
#r "RDotNet.FSharp.dll"
#r "XPlot.Plotly.dll"
#r "XPlot.GoogleCharts.dll"
#r "XPlot.GoogleCharts.Deedle.dll"
#r "Google.DataTable.Net.Wrapper.dll"
#r "Newtonsoft.Json.dll"
namespace FsLab

#if NO_FSI_ADDPRINTER
#else
module FsiAutoShow =
  open FSharp.Charting
  open RProvider

  fsi.AddPrinter(fun (printer:Deedle.Internal.IFsiFormattable) ->
    "\n" + (printer.Format()))
  fsi.AddPrinter(fun (ch:FSharp.Charting.ChartTypes.GenericChart) ->
    ch.ShowChart() |> ignore; "(Chart)")
  fsi.AddPrinter(fun (synexpr:RDotNet.SymbolicExpression) ->
    synexpr.Print())

  open System.IO
  open Foogle
  open Foogle.SimpleHttp

  let server = ref None
  let tempDir = Path.GetTempFileName()
  let pid = System.Diagnostics.Process.GetCurrentProcess().Id
  let counter = ref 1

  do File.Delete(tempDir)
  do Directory.CreateDirectory(tempDir) |> ignore

  let displayHtml html = 
    match server.Value with
    | None -> server := Some (HttpServer.Start("http://localhost:8084/", tempDir))
    | _ -> ()
    let file = sprintf "show_%d_%d.html" pid counter.Value
    File.WriteAllText(Path.Combine(tempDir, file), html)
    System.Diagnostics.Process.Start("http://localhost:8084/" + file) |> ignore
    incr counter
      
  fsi.AddPrinter(fun (chart:FoogleChart) ->
    chart
    |> Foogle.Formatting.Google.CreateGoogleChart
    |> Foogle.Formatting.Google.GoogleChartHtml
    |> displayHtml 
    "(Foogle Chart)" )

  fsi.AddPrinter(fun (chart:XPlot.GoogleCharts.GoogleChart) ->
    let ch = chart |> XPlot.GoogleCharts.Chart.WithSize (800, 600)
    ch.Html |> displayHtml
    "(Google Chart)")

  fsi.AddPrinter(fun (chart:XPlot.Plotly.Figure) ->
    let name = 
      match chart.Layout with
      | Some ly -> ly.title
      | None -> sprintf "XPlot Generated Chart %d" counter.Value
    """<!DOCTYPE html>
    <html>
    <head><title>Plotly Chart</title></head>
    <body>""" + chart.GetInlineHtml(name) + "</body></html>" |> displayHtml
    "(Plotly Chart)" )
#endif

namespace FSharp.Charting
open FSharp.Charting
open Deedle

[<AutoOpen>]
module FsLabExtensions =
  type FSharp.Charting.Chart with
    static member Line(data:Series<'K, 'V>, ?Name, ?Title, ?Labels, ?Color, ?XTitle, ?YTitle) =
      Chart.Line(Series.observations data, ?Name=Name, ?Title=Title, ?Labels=Labels, ?Color=Color, ?XTitle=XTitle, ?YTitle=YTitle)
    static member Column(data:Series<'K, 'V>, ?Name, ?Title, ?Labels, ?Color, ?XTitle, ?YTitle) =
      Chart.Column(Series.observations data, ?Name=Name, ?Title=Title, ?Labels=Labels, ?Color=Color, ?XTitle=XTitle, ?YTitle=YTitle)
    static member Pie(data:Series<'K, 'V>, ?Name, ?Title, ?Labels, ?Color, ?XTitle, ?YTitle) =
      Chart.Pie(Series.observations data, ?Name=Name, ?Title=Title, ?Labels=Labels, ?Color=Color, ?XTitle=XTitle, ?YTitle=YTitle)
    static member Area(data:Series<'K, 'V>, ?Name, ?Title, ?Labels, ?Color, ?XTitle, ?YTitle) =
      Chart.Area(Series.observations data, ?Name=Name, ?Title=Title, ?Labels=Labels, ?Color=Color, ?XTitle=XTitle, ?YTitle=YTitle)
    static member Bar(data:Series<'K, 'V>, ?Name, ?Title, ?Labels, ?Color, ?XTitle, ?YTitle) =
      Chart.Bar(Series.observations data, ?Name=Name, ?Title=Title, ?Labels=Labels, ?Color=Color, ?XTitle=XTitle, ?YTitle=YTitle)

namespace Foogle
open Deedle

[<AutoOpen>]
module FoogleExtensions =

  type Foogle.Chart with
    static member PieChart(frame:Frame<_, _>, column, ?Label, ?PieHole) =
      Foogle.Chart.PieChart
        ( frame.GetColumn<float>(column) |> Series.observations,
          ?Label=Label, ?PieHole=PieHole)
    static member GeoChart(frame:Frame<_, _>, column, ?Label, ?Region, ?DisplayMode) =
      Foogle.Chart.GeoChart
        ( frame.GetColumn<float>(column) |> Series.observations,
          ?Label=Label, ?Region=Region, ?DisplayMode=DisplayMode)

namespace MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.LinearAlgebra
open Deedle

module Matrix =
  let inline toFrame matrix = matrix |> Matrix.toArray2 |> Frame.ofArray2D
module DenseMatrix =
  let inline ofFrame frame = frame |> Frame.toArray2D |> DenseMatrix.ofArray2
module SparseMatrix =
  let inline ofFrame frame = frame |> Frame.toArray2D |> SparseMatrix.ofArray2
module Vector =
  let inline toSeries vector = vector |> Vector.toSeq |> Series.ofValues
module DenseVector =
  let inline ofSeries series = series |> Series.values |> Seq.map (float) |> DenseVector.ofSeq
module SparseVector =
  let inline ofSeries series = series |> Series.values |> Seq.map (float) |> SparseVector.ofSeq


namespace Deedle
open Deedle
open MathNet.Numerics.LinearAlgebra

module Frame =
  let inline ofMatrix matrix = matrix |> Matrix.toArray2 |> Frame.ofArray2D
  let inline toMatrix frame = frame |> Frame.toArray2D |> DenseMatrix.ofArray2

  let ofCsvRows (data:FSharp.Data.Runtime.CsvFile<'T>) =
    match data.Headers with
    | None -> Frame.ofRecords data.Rows
    | Some names -> Frame.ofRecords data.Rows |> Frame.indexColsWith names

module Series =
  let inline ofVector vector = vector |> Vector.toSeq |> Series.ofValues
  let inline toVector series = series |> Series.values |> Seq.map (float) |> DenseVector.ofSeq
