#if INTERACTIVE
#r @"Microsoft.Kinect.dll"
#else
module Kinect20
#endif

open Microsoft.Kinect
open System.Collections.Generic
let bodyFrameReader = 
    // Detect Sensor
    let kinectSensor = KinectSensor.GetDefault()
    kinectSensor.Open()
    // Open Reader to cature source
    kinectSensor.BodyFrameSource.OpenReader()

let stop() = bodyFrameReader.Dispose()

let bodyTracking = 
    bodyFrameReader.FrameArrived 
    |> Event.add(
        fun bfae ->
        // Acquire a Frame
        use bodyFrame = bfae.FrameReference.AcquireFrame()
        if bodyFrame.BodyCount > 0 then
            let bodyContainer = Array.create bodyFrame.BodyCount (Unchecked.defaultof<Body>)
            do bodyFrame.GetAndRefreshBodyData(bodyContainer)
            let parts = bodyContainer |> Seq.filter (fun b -> b.IsTracked)
            parts
            |> Seq.map (fun b -> b.Joints)
            |> Seq.iter(fun j -> j |> Seq.iter(fun p -> 
                        let part = p.Key.ToString("F")
                        let pos = p.Value.Position                              
                        printfn "Bodypart %s at %f %f %f" part pos.X pos.Y pos.Z))) 

