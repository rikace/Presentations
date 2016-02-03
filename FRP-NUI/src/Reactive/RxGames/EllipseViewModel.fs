namespace ViewModels

open System
open System.Windows
open FSharp.ViewModule
open FSharp.ViewModule.Validation
open FsXaml
open System.Reactive
open System.Reactive.Concurrency
open System.Reactive.Linq


type EllipseView = XAML<"EllipseDemo.xaml", true>

module EllipseModule = 
    let getAngleStream() : IObservable<float>=

        Observable.Generate(0., (fun _ -> true), 
                                (fun i -> (i + 1.) % 360.), 
                                (fun i -> i), 
                                (fun _ -> TimeSpan.FromMilliseconds(10.)), 
                                ThreadPoolScheduler.Instance)

    let getLocationStreamFromChunkyGenerator() : IObservable<Point> =
        Observable.Generate(
                0.,
                (fun i -> i < 40.),
                (fun i -> i + 1.),
                (fun i->
                    match int(i % 4.) with
                    | 0 -> new Point(100., 100.)
                    | 1 -> new Point(150., 100.)
                    | 2 -> Point(150., 150.)
                    | 3 -> Point(100., 150.)
                    | _ -> raise (new ArgumentException())),
                (fun i -> TimeSpan.FromMilliseconds(250.))
        )

    let getAngleStreamByInterval(interval) : IObservable<float> =
        Observable.Interval(interval, ThreadPoolScheduler.Instance)
        |> Observable.map(fun i -> float(i) % 360.)

    let getLocationStreamFromFineGrainedGenerator() :IObservable<Point>=
        Observable.Generate(
                0.,
                (fun _ -> true),
                (fun i -> (i + 1.) % 360.),
                (fun i -> 
                    new Point(
                        150. + 100. * Math.Sin(Math.PI * i / 180.),
                        150. + 100. * Math.Cos(Math.PI * i / 180.)
                        )),
                (fun _ -> TimeSpan.FromMilliseconds(10.)),
                ThreadPoolScheduler.Instance)

    let getChunkyLocationStreamFromAtoms() : IObservable<Point> =
        let wait = Observable.Empty<Point>().Delay(TimeSpan.FromMilliseconds(250.))
        let points = [| new Point(100., 100.);
                        new Point(150., 100.);
                        new Point(150., 150.);
                        new Point(100., 150.) |]


        let locs = points.ToObservable(ThreadPoolScheduler.Instance)
                             .Select(fun pt -> Observable.Return(pt).Concat(wait))
                             .Concat()
        locs.Repeat(10)

type EllipseViewModel() as self = 
    inherit ViewModelBase()    

    do
//        let locations = 
//            EllipseModule.getLocationStreamFromChunkyGenerator()

       // let locations = EllipseModule.getChunkyLocationStreamFromAtoms()
//        let locations = 
//            EllipseModule.getLocationStreamFromFineGrainedGenerator()

        // Circle high frame
//        let locations = 
//            EllipseModule.getAngleStream()
//            |> Observable.map(fun a -> Math.PI * a / 180.)
//            |> Observable.map(fun a -> new Point(150. + 100. * Math.Sin(a), 150. + 100. * Math.Cos(a)))


        let locations = 
            EllipseModule.getAngleStream()
            |> Observable.map(fun a ->  new Point(70. + a * 1.2, 200. + 150. * Math.Sin(Math.PI * a / 180.)))

        locations.SubscribeOn(ThreadPoolScheduler.Instance).Subscribe(self.SetPosition) |> ignore



    let mutable ballLeft = 80.
    let mutable ballTop = 80.

    member self.BallLeft 
        with get() = ballLeft
        and set value = 
            ballLeft <- value
            self.RaisePropertyChanged("BallLeft")


    member self.BallTop 
        with get() = ballTop
        and set value = 
            ballTop <- value
            self.RaisePropertyChanged("BallTop")    

    member self.SetPosition(position:Point) =
        self.BallLeft <- position.X
        self.BallTop <- position.Y