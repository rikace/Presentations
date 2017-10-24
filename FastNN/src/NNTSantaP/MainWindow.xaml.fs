namespace Views

open System
open FsXaml
open FSharp.Charting
open FSharp.Charting.ChartTypes
open System.Windows
open System.Windows.Forms.Integration

type MainWindowBase = XAML<"MainWindow.xaml">

type MainWindow() =
    inherit MainWindowBase()