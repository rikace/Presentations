open System
open FsXaml
open System.Windows
open System.Windows.Controls
open System.Reactive.Linq
open System.Threading
open System.Reactive.Contrib.Monitoring
open System.Threading.Tasks
open System.Diagnostics

type MainWindow = XAML<"MainWindow.xaml">

[<STAThread>]
[<EntryPoint>]
let main _ =
    Gjallarhorn.Wpf.Platform.install true |> ignore

    let app = Application()
    let win = MainWindow()

    let button1 : Button = win.button1
    let button2 : Button = win.button2
    let btnIntervalStart : Button = win.btnIntervalStart
    let btnIntervalStop : Button = win.btnIntervalStop
    let btnBufferStart : Button = win.btnBufferStart
    let btnBufferStop : Button = win.btnBufferStop
    let btnWindowStart : Button = win.btnWindowStart
    let btnWindowStop : Button = win.btnWindowStop
    let btnGroupByStart : Button = win.btnGroupByStart
    let btnGroupByStop : Button = win.btnGroupByStop

    button1.Click
    |> Event.merge button2.Click
    |> Event.add (fun a -> MessageBox.Show "Hii from event"
                           |> ignore)

    button1.Click
    |> Observable.merge button2.Click
    |> Observable.add (fun a -> MessageBox.Show "Hii from observable"
                                |> ignore)

    VisualRxSettings.ClearFilters()

    let info = VisualRxSettings.Initialize(VisualRxWcfDiscoveryProxy.Create())

    let infos = info.Result
    Trace.WriteLine(infos)

    let subscribe (btnStart : Button) (btnStop : Button) (observable : IObservable<_>) (action : _ -> unit) =
        btnStart.Click
        |> Event.add (fun a ->
            let subscription = observable.Subscribe action
            btnStart.Tag <- subscription
            ())

        btnStop.Click
        |> Event.add (fun a ->
            let subscription = btnStart.Tag :?> IDisposable
            subscription.Dispose()
            ())

    let interval =
        Observable
            .Interval(TimeSpan.FromSeconds(1.0))
            .Monitor("Interval", 1.0, [||])
    subscribe btnIntervalStart btnIntervalStop interval (printfn "Interval %d")

    let buffer =
        Observable
            .Interval(TimeSpan.FromSeconds(1.0))
            .Buffer(3)
            .Select(fun xs -> String.Join(", ", xs))
            .Monitor("Buffer", 1.0, [||])
    subscribe btnBufferStart btnBufferStop buffer (printfn "Buffer %A")

    let window =
        Observable
            .Interval(TimeSpan.FromSeconds(1.0))
            .Window(3)
            .MonitorMany("Window", 1.0, [||])
    subscribe btnWindowStart btnWindowStop window (fun obs -> obs.Sum().Subscribe (printfn "Window %d") |> ignore)

    let groupBy =
        Observable
            .Interval(TimeSpan.FromSeconds(1.0))
            .GroupBy(fun x -> x % 3L)
            .MonitorGroup("GroupBy", 1.0, [||])
    let printKeyValue key value =
        printfn "GroupBy %s %d" (String('*', key)) value
        ()
    subscribe btnGroupByStart btnGroupByStop groupBy (fun obs -> obs.Subscribe (printKeyValue (int obs.Key)) |> ignore)

    let txt : TextBox = win.txt
    let console : TextBlock = win.console

    let observable =
        Observable
            .FromEventPattern(txt, "TextChanged")
            .Monitor("TextChanged", 1.0, [||])
            .Select(fun e -> txt.Text)
            .Monitor("Select", 2.0, [||])
            .Throttle(TimeSpan.FromSeconds(1.0))
            .Monitor("Throttle", 4.0, [||])
            .Where(fun e -> e.Length > 3)
            .Monitor("Where", 3.0, [||])
            .DistinctUntilChanged()
            .Monitor("AutoComplete", 5.0, [||])

    let subscription =
        observable
            .ObserveOn(SynchronizationContext.Current)
            .Subscribe (fun text -> console.Text <- console.Text + Environment.NewLine + text )

    app.Run win
