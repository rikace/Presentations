namespace RxGames

/// Encapsulates a triggerable instance
type ITriggerable<'a> =
    abstract member Trigger : 'a -> unit
    
/// Encapsulates an event that is both triggerable and observable    
type IHappen<'a> = 
    inherit ITriggerable<'a> 
    inherit System.IObservable<'a>
            
/// Encapsulates repository of IHappen instances
type IHappenings =
    abstract member ObtainHappening<'a> : unit -> IHappen<'a>
   
/// Repository of IHappen instances      
type Happenings () =
    let happenings = System.Collections.Generic.Dictionary<System.Type,_>()
    let happeningsLock = obj()
    interface IHappenings with
        member this.ObtainHappening<'a> () =            
            let CreateHappening () = 
                let observers = ref []
                let add observer = 
                    observers := 
                        observer::!observers
                let remove observer = 
                    observers := 
                        !observers |> List.filter(fun x -> observer.Equals(x))
                let trigger x =
                    !observers 
                    |> List.iter (fun (observer:System.IObserver<'a>) -> 
                        observer.OnNext(x))
                let happenLock = obj()                        
                { new IHappen<'a> with                        
                    member this.Subscribe (observer:System.IObserver<'a>) =                                                                                    
                        lock happenLock (fun _ -> add observer)
                        { new System.IDisposable with
                            member this.Dispose() = 
                                lock happenLock (fun _ -> remove observer)                                
                        }                               
                    member this.Trigger (x:'a) = trigger x                       
                }               
            lock happeningsLock (fun _ ->    
                match happenings.TryGetValue(typeof<'a>) with
                | true, happen -> unbox happen
                | false, _ ->                 
                    let happen = CreateHappening ()
                    happenings.Add(typeof<'a>,box happen)
                    happen
            )                    
    end
      
/// Functions operating on IObservable<T>
module Observable = 
    open System
    open System.Windows.Threading
        
    /// Observer helper lifted from lib\FSharp.Core\control.fs
    [<AbstractClass>]  
    type BasicObserver<'a>() =
        let mutable stopped = false
        abstract Next : value : 'a -> unit
        abstract Error : error : exn -> unit
        abstract Completed : unit -> unit
        interface IObserver<'a> with
            member x.OnNext value = 
                if not stopped then x.Next value
            member x.OnError e = 
                if not stopped then stopped <- true
                x.Error e
            member x.OnCompleted () = 
                if not stopped then stopped <- true
                x.Completed ()
    
    /// Tap into a sequence of Observable expressions
    let tap f (w:IObservable<_>) =
        let hook (observer:IObserver<_>) =
            { new BasicObserver<_>() with  
                member x.Next(v) = 
                    match (try f v; None with | exn -> Some(exn)) with
                    | Some(exn) -> observer.OnError exn
                    | None -> observer.OnNext v                    
                member x.Error(e) = 
                    observer.OnError(e)
                member x.Completed() = 
                   observer.OnCompleted() 
            } 
        { new IObservable<_> with 
            member x.Subscribe(observer) =
                w.Subscribe (hook(observer))                    
        }
   
    /// Invoke Observer function through specified function
    let invoke f (w:IObservable<_>) =
        let hook (observer:IObserver<_>) =
            { new BasicObserver<_>() with  
                member x.Next(v) = 
                    f (fun () -> observer.OnNext v)
                member x.Error(e) = 
                    f (fun () -> observer.OnError(e))
                member x.Completed() = 
                    f (fun () -> observer.OnCompleted()) 
            } 
        { new IObservable<_> with 
            member x.Subscribe(observer) =
                w.Subscribe (hook(observer))
        }
 
    /// Delay execution of Observer function
    let delay milliseconds (observable:IObservable<'a>) =
        let f g =
            async {
                do! Async.Sleep(milliseconds)
                do g ()
            } |> Async.Start
        invoke f observable 
     
    /// Execture Observer function on Dispatcher thread
    /// <Remarks>For WPF and Silverlight</remarks> 
    let onDispatcher (observable:IObservable<'a>) =
        let dispatcher = Dispatcher.CurrentDispatcher
        let f g =
            dispatcher.BeginInvoke(Action(fun _ -> g())) |> ignore
        invoke f observable
          
module Test =  
    open System
    open System.Windows
    open System.Windows.Controls
    open System.Windows.Input
    open System.Windows.Media
    open System.Windows.Threading           
        
    let getPosition (element : #UIElement) (args : MouseEventArgs) =
        let point = args.GetPosition(element)
        (point.X, point.Y)    
     
    type TimeFliesWindow(happenings:IHappenings) as this =
        inherit Window()        

        let canvas = Canvas(Width=800.0,Height=400.0,Background=Brushes.White) 
        do this.Content <- canvas       

        do "F# can react to second class events!"
            |> Seq.iteri(fun i c ->  
                let s = TextBlock(Width=20.0, 
                                Height=30.0, 
                                FontSize=20.0, 
                                Text=string c, 
                                Foreground=Brushes.Black, 
                                Background=Brushes.White)
                canvas.Children.Add(s) |> ignore              
                
                happenings.ObtainHappening<MouseEventArgs>()                                 
                |> Observable.map (getPosition canvas)       
                |> Observable.tap (fun p -> Console.WriteLine p)
                |> Observable.delay (i * 100)
                |> Observable.onDispatcher
                |> Observable.subscribe (fun (x, y) ->                                                        
                     Canvas.SetTop(s, y) 
                     Canvas.SetLeft(s, x + float ( i * 10)))              
                |> ignore
           )        
              
    let happenings = new Happenings() :> IHappenings
    let win = TimeFliesWindow(happenings,Title="Time files like an arrow")   

    do  // Publish mouse movement event arguments
        let happen = happenings.ObtainHappening<MouseEventArgs>()
        win.MouseMove         
        |> Observable.subscribe happen.Trigger |> ignore

    [<STAThread>]
    do (new Application()).Run(win) |> ignore                  