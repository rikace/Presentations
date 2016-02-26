namespace KinectFS.Core

open System
open System.Linq
open System.Reactive
open System.Reactive.Linq
open Microsoft.Kinect
open RxFsharp

[<AutoOpen>]
module KinectLib =

    //  For me, Reactive Extensions is a natural fit for Kinect programming in an
    // event driven application as it removes the need to think about the mechanics 
    // of how to compose and orchestrate the associated event streams. A similar 
    // example to that above but using Rx shows how to subscribe to an Observable 
    // and shows how to specify which threading context the subscription and event
    //  delivery should occur on.
    
    type KinectSensor with
        member this.Apply(f) = 
            let reader = this.BodyFrameSource.OpenReader()
            let obs  =
                reader.FrameArrived
                |> Observable.subscribe(fun x -> 
                        use bodyRef = x.FrameReference.AcquireFrame()
                        if bodyRef <> null then
                            f bodyRef)
            { new IDisposable with 
                member x.Dispose() =
                    obs.Dispose()
                    reader.Dispose() }
                    
            

    // IObservable extension methods to extend the Kinect API with the ReactiveExtensions programming model
    type KinectSensor with
        member this.GetAllFramesReadyObservable() =
            // get hold of a reader for the frames of body frames that come from the sensor
            let reader = this.BodyFrameSource.OpenReader()            
            { new IObservable<Body array> with
                    member x.Subscribe(observer) = 
                        let obser = 
                            reader.FrameArrived
                            //The “problem” is that this delivers a set of BodyFrameArrivedEventArgs but in order to make use of them I need to get hold of the BodyFrame itself calling AcquireFrame() and that sometimes returns NULL
                            |> Observable.map(fun f ->f.FrameReference.AcquireFrame()) 
                            |> Observable.filter(fun f -> f <> null)
                            |> Observable.subscribe(fun frame -> 

                            // return the shared Body[] for all frames reducing the complexity it might introduce around concurrent access and so on or whether I wanted to try and allocate a new Body[] array with the arrival of each frame with all the problems it might introduce around memory pressure.

                                        let bodies = Array.zeroCreate<Body> frame.BodyCount   
                                        frame.GetAndRefreshBodyData(bodies)
                                        frame.Dispose()
                                        observer.OnNext(bodies))
                        { new IDisposable with 
                            member x.Dispose() = 
                                    obser.Dispose()
                                    reader.Dispose() } 
                                                        }      

    type KinectSensor with
        member this.GetColorFrameReadyObservable() =
            let reader = this.ColorFrameSource.OpenReader()            
            { new IObservable<ColorFrameArrivedEventArgs> with
                    member x.Subscribe(observer) = 
                        let obser = reader.FrameArrived |> Observable.subscribe(fun ev -> observer.OnNext(ev))
                        { new IDisposable with 
                            member x.Dispose() = 
                                    obser.Dispose()
                                    reader.Dispose() } }   

module MiscKinect =
    
    type HandPosition = {Id:uint64;Left:CameraSpacePoint;Right:CameraSpacePoint}
    type HandDistance = {Id:uint64;Distance:float}

    let obsBodies (kinectSensor:KinectSensor) =
         kinectSensor.GetAllFramesReadyObservable()
         |> Observable.flatmapSeq(fun f -> f |> Array.toSeq)         
         |> Observable.filter(fun f -> f.IsTracked)
    
    let getCameraSpacePoint(cameraSpacePoint:CameraSpacePoint) =
            (cameraSpacePoint.X,cameraSpacePoint.Y,cameraSpacePoint.Z)

    let leftORroghtHandPositions (kinectSensor:KinectSensor) =
        kinectSensor
        |> obsBodies
        |> Observable.filter(fun f -> f.Joints.[JointType.HandLeft].TrackingState <> TrackingState.NotTracked && f.Joints.[JointType.HandRight].TrackingState <> TrackingState.NotTracked)
        |> Observable.map(fun f -> 
                {   Id = f.TrackingId 
                    Left = f.Joints.[JointType.HandLeft].Position
                    Right = f.Joints.[JointType.HandRight].Position } )


    let cameraSpaceDistance(p1:CameraSpacePoint) (p2:CameraSpacePoint) =
        Math.Sqrt(Math.Pow(float p1.X - float p2.X, 2.) + Math.Pow(float p1.Y - float p2.Y, 2.) + Math.Pow(float p1.Z - float p2.Z, 2.))


    let distanceBetweenHands (kinectSensor:KinectSensor) =
        kinectSensor
        |> obsBodies
        |> Observable.filter(fun f -> f.Joints.[JointType.HandLeft].TrackingState <> TrackingState.NotTracked && f.Joints.[JointType.HandRight].TrackingState <> TrackingState.NotTracked)
        |> Observable.map(fun f -> 
                let leftHandPosition = f.Joints.[JointType.HandLeft].Position
                let righHandPosition = f.Joints.[JointType.HandRight].Position
                {   Id = f.TrackingId 
                    Distance = cameraSpaceDistance leftHandPosition righHandPosition })

    let clap (kinectSensor:KinectSensor) =
        let obsHands =kinectSensor |> distanceBetweenHands
        let obsHandPrevious = 
            obsHands
            |> Observable.skipSpan(TimeSpan.FromSeconds(1.))
        obsHands
        |> Observable.zip(obsHandPrevious)
        // the 0.75 and 0.05 are just arbitrary values I got by playing around with the sensor in the configuration 
        |> Observable.filter(fun (a,b) -> a.Distance > 0.75 && b.Distance < 0.05)

    let swipe (kinectSensor:KinectSensor) =
        let obsHands = kinectSensor |> obsBodies
        let obsRightHand = 
            obsHands
            |> Observable.filter(fun f -> f.Joints.[JointType.HandRight].TrackingState <> TrackingState.NotTracked)
            |> Observable.map(fun f -> float f.Joints.[JointType.HandRight].Position.X)
        let obsightHandPrevious = 
            obsRightHand
            |> Observable.skipSpan(TimeSpan.FromSeconds(1.))
        let obsHandZipped =
            obsRightHand
            |> Observable.zip(obsightHandPrevious)
            |> Observable.filter(fun (a,b) -> a < -0.3 && b > 0.3)
        obsHandZipped

    let jumpimg (kinectSensor:KinectSensor) =
        let feetObs = 
            kinectSensor 
            |> obsBodies
            |> Observable.map(fun f -> (float f.Joints.[JointType.FootLeft].Position.Y + float f.Joints.[JointType.FootRight].Position.Y) / 2.)
        // This extracts the average vertical position of the left and right foot. 
        // This is a simplification as it would be possible to trick the algorithm 
        // by just lifting one foot up twice as high as both feet would need to jump up. 
        Observable.Buffer(feetObs, TimeSpan.FromSeconds(1.), TimeSpan.FromMilliseconds(200.))
        // This is the actual jump detection code. The idea is that to have jumped, the player would have to have his feet low, then high and then low again in a short timespan. 
        // To analyze this, we’re looking at samples from a time frame of one second
        // After this, we’ll move forward by 200 milliseconds and analyze again. This magic is provided by the Buffer() extension method. 
        // We’ll look for the maximum height of the feet within the time frame and see if the first and last samples are both lower than the maximum minus a hard-coded magic number (for simplification again). 
        // If the algorithm matches, the IObservable outputs Some, otherwise None
        |> Observable.filter(fun f -> f.Count > 2)
        |> Observable.map(fun f -> 
                let lowThreashold = f |> Seq.max
                if (f |> Seq.head) < lowThreashold && (f |> Seq.last) > lowThreashold then Some()
                else None)


//
//void sensor_SkeletonFrameReady(object sender, SkeletonFrameReadyEventArgs e)
//        {
//            try
//            {
//                using (var skeletonFrame = e.OpenSkeletonFrame())
//                {
//                    if (skeletonFrame == null)
//                        return;
//
//                    if (skeletons == null ||
//                        skeletons.Length != skeletonFrame.SkeletonArrayLength)
//                    {
//                        skeletons = new Skeleton[skeletonFrame.SkeletonArrayLength];
//                    }
//                    skeletonFrame.CopySkeletonDataTo(skeletons);
//                }
//                Skeleton closestSkeleton = skeletons.Where(s => s.TrackingState == SkeletonTrackingState.Tracked)
//                                                    .OrderBy(s => s.Position.Z * Math.Abs(s.Position.X))
//                                                    .FirstOrDefault();
//                if (closestSkeleton == null)
//                    return;
//
//                var rightFoot = closestSkeleton.Joints[JointType.FootRight];
//                var leftFoot = closestSkeleton.Joints[JointType.FootLeft];
//                var rightHand = closestSkeleton.Joints[JointType.HandRight];
//                var leftHand = closestSkeleton.Joints[JointType.HandLeft];
//
//                if (rightFoot.TrackingState == JointTrackingState.NotTracked ||
//                    rightHand.TrackingState == JointTrackingState.NotTracked ||
//                    leftHand.TrackingState == JointTrackingState.NotTracked)
//                {
//                    //Don't have a good read on the joints so we cannot process gestures
//                    return;
//                }
//
//                CoordinateMapper mapper = sensor.CoordinateMapper;
//                var point = mapper.MapSkeletonPointToColorPoint(closestSkeleton.Joints[JointType.Head].Position, sensor.ColorStream.Format);
//                var point1 = mapper.MapSkeletonPointToColorPoint(rightHand.Position, sensor.ColorStream.Format);
//                var point2 = mapper.MapSkeletonPointToColorPoint(leftHand.Position, sensor.ColorStream.Format);
//                    // - Put Your Draw Code Here insted of the following:
//                    SetEllipsePosition(ellipseRightFoot, rightFoot, false);
//                    SetEllipsePosition(ellipseLeftFoot, leftFoot, false);
//                    SetEllipsePosition(ellipseLeftHand, leftHand, isBackGestureActive);
//                    SetEllipsePosition(ellipseRightHand, rightHand, isForwardGestureActive);
//                    SetImagePosition(punhal, rightHand);
//                    SetImagePosition2(punhal2, leftHand);
//                    // -------------------------------------------------
//            catch
//            {
//                myException(this, new EventArgs());
//            }
//        }
//                

    List.map2 (fun x y -> x+y)  [1;2]  [3;4]


    let (<+>) fctList list = List.map2 (fun fct elem -> fct elem) fctList list
    let r = List.map (fun x y z -> x+y+z) [1;2] <+> [1;2] <+> [1;2]

    let map4 f l1 l2 l3 l4 = List.map f l1 <+> l2 <+> l3 <+> l4


module KinectTest =

    let toDifference prev x = 
        let res = abs(x - prev)
        res

    let kinect = KinectSensor.GetDefault()

    let obsBodies (kinectSensor:KinectSensor) =
         kinectSensor.GetAllFramesReadyObservable()
         |> Observable.flatmapSeq(fun f -> f |> Array.toSeq)         
         |> Observable.filter(fun f -> f.IsTracked)

    kinect
    |> obsBodies
    |> Observable.map(fun x -> x.Joints.[JointType.HandRight].Position.X)
    |> Observable.map(toDifference(float32 0))
    |> fun x -> Observable.Buffer(x, 10, 1)
    |> Observable.map Enumerable.Max