namespace Threading
//#light
//
//#r "WindowsBase"
//#r "PresentationCore"
//#r "PresentationFramework"

    open System
    open System.Windows
    open System.Windows.Controls
    open System.Threading
    open System.Windows.Threading

    module srcModule = 

        // create a reference to a WPF control in interactive window
        let mutable (wp : TextBlock) = null

        // create a new WPF gui thread with a running dispatcher and message pump
        let thread = new System.Threading.Thread(fun() -> 
            let window = new System.Windows.Window(Name="Test",Width=500.0,Height=500.0)
            wp <- new TextBlock()
            wp.Text <- "test1"
            window.Content <- wp
            window.Visibility <- Visibility.Visible
            window.Show()
            window.Closed.Add(fun e -> 
                Dispatcher.CurrentDispatcher.BeginInvokeShutdown(DispatcherPriority.Background)
                Thread.CurrentThread.Abort())
            Dispatcher.Run()
            )
        thread.SetApartmentState(ApartmentState.STA)
        thread.IsBackground <- true

        // start the thread, which will invoke the popup ui
        thread.Start()

        // once WPF window is up, you can marshall updates via its running dispatcher
        wp.Dispatcher.BeginInvoke(Action(fun _ -> wp.Text <- "test2")) |> ignore


