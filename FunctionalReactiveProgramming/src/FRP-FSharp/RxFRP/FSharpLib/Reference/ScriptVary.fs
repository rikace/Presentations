namespace Utility

open System
open System.IO
open System.Windows
open System.Threading
open System.Windows.Threading

module srcModule = 

    let binaryofint n =
        [ for i in 8 * sizeof<int> - 1 .. -1 .. 0 ->
            if ( n >>> i) % 2 = 0 then "0" else "1" ]
        |> String.concat ""
   
    let readBytesOfFile filename =
        seq {   use stream = File.OpenRead filename
                let input = ref(stream.ReadByte())
                while !input <> -1 do
                    yield byte !input
                    input := stream.ReadByte() }    

    let readLinesOfFile filename =
        seq {   use stream = File.OpenRead filename
                use reader = new StreamReader(stream)
                while not reader.EndOfStream do
                    yield reader.ReadLine() }            


//    binaryofint 10
//    readBytesOfFile @"C:\Users\Riccardo\Desktop\Human Rig\04_Arm.jpg"                
//    readLinesOfFile @"H:\Temp\words.txt"    


    //I# "c:\Program Files\Reference Assemblies\Microsoft\Framework\v4.0"

    //#r "System.Core.dll"
    //#r "WindowsBase.dll"
    //#r "PresentationFramework.dll"
    //#r "PresentationCore.dll"


    let makeWindow() =
        let w = Window()
        w.Title <- "Sample"
        w.Width <- 500.
        w.Height <- 500.
        w.Content <- "Content"
        w.Show()
        w
    //makeWindow.Close()    


    let ui =
        let mk() =
            let w = new ManualResetEvent(false)
            let application = ref null
            let start() =
                let app = Application()
                application := app
                ignore(w.Set())
                app.Run() |> ignore
            let t = Thread start
            t.IsBackground <- true
            t.SetApartmentState ApartmentState.STA
            t.Start()
            ignore(w.WaitOne())
            !application, t
        lazy(mk())       
    
    let spanw : ('a -> 'b) -> 'a -> 'b =
        fun f x ->
            let application, thread = ui.Force()
            let f _ =
                try (fun f_x () -> f_x) (f x) with e -> (fun () -> raise e)
            let t = application.Dispatcher.Invoke(DispatcherPriority.Send, System.Func<_, _>(f), null)
            (t :?> unit -> 'b)()


    type Scene() =
        inherit UIElement()
    
        override this.OnRender dc =
            base.OnRender dc
            let pen = Media.Pen(Media.Brushes.Red, 0.3)
            for i=0 to 100 do
                let x = 4. * float i
                let xys = [x, 0.; 400., x; 400. - x, 400.; 0., 400. - x; x, 0.]
                for (x0, y0), (x1, y1) in Seq.pairwise xys do
                    dc.DrawLine(pen, Point(x0, y0), Point(x1, y1))

    spanw (fun () -> makeWindow().Content <- "CIAO") ()


    let rec fib = function
        | 0 | 1 as n -> n
        | n -> fib(n-2) + fib(n-1)

    let rec fibAsync = function
        | 0 | 1 as n -> async {return n}
        | n -> async {  let! f2 = fibAsync(n-2)
                        let! f1 = fibAsync(n-1)
                        return f1 + f2 }

    let rec fibAsyncChild = function
        | 0 | 1 as n -> async {return n}
        | n -> async {  let! f2 = fibAsync(n-2) |> Async.StartChild
                        let! f1 = fibAsync(n-1)
                        let! resF2 = f2
                        return f1 + resF2 }



