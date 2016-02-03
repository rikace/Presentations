module AirplaneListener

open System
open Leap

type SwipeDirection =
    | Up 
    | Down 
    | Left 
    | Right

type SwipeActionRec = {Fingers:FingerList;SwipeDirection:SwipeDirection}


type IAirplaneViewModelInterface =
    
    abstract XAngle : double with get,set
    abstract YAngle : double with get,set



type LeapListener() =
    inherit Listener()

    let leapSwipeEvent = Event<SwipeActionRec>()

    let zeroFLoat = float32 0

    let swipeAction (swipeActionRec:SwipeActionRec) =
        let fingersCount = swipeActionRec.Fingers.Count
        if fingersCount = 5 then leapSwipeEvent.Trigger swipeActionRec

    let leapAgent = MailboxProcessor<FingerList * GestureList>.Start(fun inbox ->
        let rec loop() = async {
            let! fingers, gestures = inbox.Receive()
            for gst:Gesture in gestures do
                let swipe:SwipeGesture = new SwipeGesture(gst)
                if  Math.Abs(swipe.Direction.x) > Math.Abs(swipe.Direction.y) then
                    if swipe.Direction.x > zeroFLoat then // right swipe
                        swipeAction {Fingers = fingers; SwipeDirection = SwipeDirection.Right }
                    else // left swipe
                        swipeAction {Fingers = fingers; SwipeDirection = SwipeDirection.Left }// Vertical swipe
                else
                    if swipe.Direction.y > zeroFLoat then // upward swipe
                        swipeAction {Fingers = fingers; SwipeDirection = SwipeDirection.Up }
                    else // downward swipe
                        swipeAction {Fingers = fingers; SwipeDirection = SwipeDirection.Down }
            return! loop() }
        loop())

    member x.LeapSwipeEvent = leapSwipeEvent.Publish


//    The OnConnect() callback method is called when your application connects to the Leap. 
//    This is the method where you enable detection of swipe-gestures and 
//    set the minimum length and velocity required to detect this gesture 
//    (The length and velocity values are in mm and mm/s respectively). 
//    Since I want to detect slight hand/finger movements as swipe-gestures
//    I have set the values of the minimum length and velocity to very small values, 10mm and 100mm/s.
    override x.OnConnect(ctlr:Controller) =         
            ctlr.Config.SetFloat("Gesture.Swipe.MinLength", float32(10)) |> ignore
            ctlr.Config.SetFloat("Gesture.Swipe.MinVelocity", float32(100)) |> ignore
            ctlr.Config.Save() |> ignore
            ctlr.EnableGesture(Gesture.GestureType.TYPESWIPE)


//    The OnFrame() callback method is called when the Leap generates a new frame of motion tracking data. 
//    This is where I process the data from the Leap to determine the direction of a swipe gesture.
    override x.OnFrame(ctrl:Controller ) =
        let currentFrame:Frame = ctrl.Frame()

        if not(currentFrame.Hands.IsEmpty) then

            let firstHand:Hand = currentFrame.Hands.[0]
            let fingers:FingerList  = firstHand.Fingers

            if not(fingers.IsEmpty) then
                let gestures:GestureList = currentFrame.Gestures()
                leapAgent.Post (fingers,gestures)




type AirplaneRotation(airplane:IAirplaneViewModelInterface) =

    let listener:LeapListener = new LeapListener()
    let ctrl:Controller=new Controller()
    let [<Literal>] shift : double = 1.

    let swipeAction(sd:SwipeActionRec) = 
        let direction = sd.SwipeDirection
        match direction with
        | Up    -> airplane.XAngle <- airplane.XAngle + shift
        | Down  -> airplane.XAngle <- airplane.XAngle - shift
        | Left  -> airplane.YAngle <- airplane.YAngle + shift  
        | Right -> airplane.YAngle <- airplane.YAngle - shift


    do
        ctrl.AddListener(listener) |> ignore
        listener.LeapSwipeEvent
        |> Observable.subscribe swipeAction
        |> ignore

    interface IDisposable with
        member x.Dispose() =
            ctrl.RemoveListener(listener) |> ignore
            ctrl.Dispose()

