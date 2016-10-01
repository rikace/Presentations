module MainApp

open System
open System.Windows
open System.Windows.Controls
open System.Windows.Markup
open FSharpWpfMvvmTemplate.ViewModel

// Create the View and bind it to the View Model
let mainWindowViewModel = Application.LoadComponent(
                             new System.Uri("/App;component/mainwindow.xaml", UriKind.Relative)) :?> Window
let viewModel = MainWindowViewModel()

mainWindowViewModel.DataContext <- viewModel

mainWindowViewModel.Loaded.AddHandler(fun o revt -> viewModel.Start() |> ignore)

// Application Entry point
[<STAThread>]
[<EntryPoint>]
let main(_) = (new Application()).Run(mainWindowViewModel)