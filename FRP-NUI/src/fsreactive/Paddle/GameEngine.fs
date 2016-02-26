namespace PongGame

module GameEngine = 
    open System
    open FsFRPLib
    open FsFRPLib.Core
    open PongGame.Rendering
    open Leap
    open Common.Random
    open PongGame.PongGameInitializer
    open Microsoft.Xna.Framework
    open Microsoft.Xna.Framework.Graphics
    open Microsoft.Xna.Framework.Input
    
    let paddleSize = 0.2
    let paddleY = -0.95
    let ballRadius = 0.05
  
    
    type StateGame = 
        { xCoordBall : float
          yCoordBall : float
          xCoordLeap : float }
    
    type GameInfo = 
        | State of StateGame
        | GameOver
    
    type Agent<'T> = MailboxProcessor<'T>





    module ``QuickLook just for refernce`` = 








        type Time = float
   
        type 'a Behavior = 
            | Behavior of (Time -> ('a * ReactBeh<'a>))
        and 'a ReactBeh = unit -> 'a Behavior   

    
        type 'a Event = 
            | Event of (Time -> (Option<'a> * ReactEvent<'a>))
        and 'a ReactEvent = unit -> 'a Event 


        
        let rec (<*>) (Behavior (behA:(Time -> ('a -> 'b) * (ReactBeh<('a -> 'b)>)))) 
                      (Behavior (behB:(Time -> 'a * (ReactBeh<'a>)))) =
            let behFun (time : Time) = 
                let (value : 'a, newBehavior : unit -> 'a Behavior) = behB time
                let (rFun : 'a -> 'b, nbfun : unit -> ('a -> 'b) Behavior) = behA time
                (rFun value, fun () -> nbfun() <*> newBehavior())
            Behavior behFun






    module LeapMotion =
     
      type LeapListenerMessage =
        | Put of Controller
        | Get of AsyncReplyChannel<float>


      type LeapListener() =
        inherit Leap.Listener()


        let leapAgent = Agent<LeapListenerMessage>.Start(fun inbox ->
                let rec loop point = async {
                    let! msg = inbox.Receive()
                    match msg with
                    | Put(ctrl) -> 
                        // right hand only one finger move
                        use frame = ctrl.Frame()
                        let hand = frame.Hands.Rightmost                
                        // 2D drawing coordinate systems put the origin at the top, left corner of the window
                        // naturally don’t use a z-axis. 
                        // this code maps Leap Motion coordinates to such a 2D system 
                        let finger = hand.Pointables.[0].StabilizedTipPosition
                        let iBox = frame.InteractionBox
                        let normalizedPoint = iBox.NormalizePoint(finger, true)
                        let x = float normalizedPoint.x //* float windowWidth
                        //let y = (1. - float normalizedPoint.y) //* float windowHeight
                        return! loop x
                    | Get(reply) -> reply.Reply(point)
                                    return! loop point }
                loop 0.)
    
        member __.GetPosition() = leapAgent.PostAndReply(fun ch -> Get ch)

        override __.OnFrame(ctrl:Controller) =
            leapAgent.Post(Put ctrl)



      type LeapListenerSingleton private() =
         static let controller = new Controller()
         static let listener = new LeapListener()
         static let leapState() = 
            controller.AddListener(listener) |> ignore
            listener

         static let mutable instance = lazy(leapState())
         static member Instance with get() = instance

      // unit -> (float * float) option
      let leapPos (game:Game) () = 
            let getPoint = LeapListenerSingleton.Instance.Value.GetPosition
            let clientBounds =  game.Window.ClientBounds
            let xm = 
                let ms = getPoint() 
                (ms * float clientBounds.Width)
            let ww = (float)clientBounds.Width
            let xmo = 2.0*(xm)/ww-1.0
            if xmo < -1.0 || 1.0 < xmo 
            then 0. else xmo


    let gameEngine (game : Game) = 










        // (float -> bool) Behavior 
        // predicate "is ball in the horizontal range game window" 
        let conditionXposition = pureBehavior (fun x -> x - ballRadius <= -1.0 || x + ballRadius >= 1.0)
   
        // (float -> float -> float -> bool) Behavior
        // predicate "is ball in the vertical range game window"  
        // or hit the paddle. 
        let conditionYposition = 
            pureBehavior 
                (fun x y xLeap -> 
                    y + ballRadius >= 1.0 
                    || ((xLeap - paddleSize/2.) <= x 
                    && x <= (xLeap + paddleSize/2.) 
                    && y <= (paddleY + ballRadius)))
   
        // (float -> bool) Behavior
        // if the range is negative then "Game Over"
        let conditionalGameOver = pureBehavior ((<=) (-1.0))
    
        // unit -> float 
        // function to update current Hand position (Leap Coordinate)
        let leapPos = LeapMotion.leapPos game


        // Event stream Hand position, hand position is treated as discrete but... I could use `accumBehavior`
        // accumBehavior starts with an initial value, an Event and create a Behavior
        // the Event becomes a producer of values that are used to update the value of the Behavior
        let ``Bad approach`` : Behavior<float> =
                let rec leapPosEvt = Event(fun _ -> (Some (leapPos()), fun () -> leapPosEvt))
                // where accumBehavior :: 'a -> 'a Event -> 'a Behavior
                accumBehavior 0.0 leapPosEvt


        // Treating event Leap position as Behavior according with the true FRP model
        let rec leapPosBeh : Behavior<float> = Behavior(fun time -> (leapPos(), (fun () -> leapPosBeh)))
        

        // float Behavior
        // Behavior of the X paddle (Leap) coordinate
        let leapPosXBehavior : Behavior<float> = 
            pureBehavior (fun x -> 
                if x - (paddleSize /2.) < -1.0 then -1.0
                elif x - (paddleSize /2.) > 1.0 then 1.0
                else x) <*> leapPosBeh
        

        // Time -> float -> float -> float Behavior -> State of Game Behavior
        // function that keep track of the current state of the Game
        let gameEngineState time coordBallX coordBallY leapPosX (* paddle X position Bevahior*) = 
          
            // float Behavior * (float -> unit)
            // where createBehavior :: 'a -> ('a Behavior * ('a -> unit))
            // createBehavior is an helper function for "lazily" re-compute the behavior on demand
            // "Reactivity" inspiration from Push-Pull paper
            let coordballUpdatedX = createBehavior coordBallX
            let coordballUpdatedY = createBehavior coordBallY
        
            // onceBehavior :: bool Behavior -> unit Event
            // where   update :: 'a Event -> 'b -> 'b Event
            // Bounce Event when the ball hit either left or right wall
            let condBounceEventX : (float -> float) Event = 
                update (onceBehavior (conditionXposition <*> (fst coordballUpdatedX))) (id)
        
            // float Behavior
            // where   accumMap :: 'a -> Event<('a -> 'b)> -> Behavior<'a>
            // Behavior value of the X speed of the ball 
            // uses accum function with initial value - producer fo values 
            let velocityBallX : float Behavior = accumMap (0.3) condBounceEventX

            // onceBehavior :: bool Behavior -> unit Event
            // where   update :: 'a Event -> 'b -> 'b Event
            // Bounce Event when the ball hit the up-down wall or paddle
            let condBounceEventY : (float -> float) Event = 
                update (onceBehavior (conditionYposition <*> (fst coordballUpdatedX) <*> (fst coordballUpdatedY) <*> leapPosX)) (id)

            // float Behavior
            // where   accumMap :: 'a -> Event<('a -> 'b)> -> Behavior<'a>
            // Behavior value of the Y speed of the ball
            // uses accum function with initial value negative becasue the ball is falling
            let velocityBallY : float Behavior = accumMap (-0.45) condBounceEventY
            
            // float Behavior
            // where computeBahavior :: 'a Behavior -> 'a Behavior
            // where bindBehaviors :: 'a Behavior -> ('b Behavior * ('a -> unit)) -> 'a Behavior
            // where integrate : float Behavior -> Time -> float -> float Behavior
            // Velocity Behavior -> Time -> Latest Coord Ball X -> 
            let updatedCoordinateBallX = 
                // Integrate is conventional name 
                //           is a physical equation that describe the position of a mass under the influence of an accelerating force
                // the speed can be positive or negative according with the direction 
                let ballXspeedUpdated : float Behavior = integrate velocityBallX time coordBallX
                // bind X speed and new X coordinate
                let currentBallXstate : float Behavior = bindBehaviors ballXspeedUpdated coordballUpdatedX
                computeBehavior currentBallXstate


            let updatedCoordinateBallY = 
                computeBehavior (bindBehaviors (integrate velocityBallY time coordBallY) coordballUpdatedY)
            

            let gameInfo : GameInfo Behavior = 
                (pureBehavior (fun xb yb xl -> 
                     State { xCoordBall = xb
                             yCoordBall = yb
                             xCoordLeap = xl })) <*> updatedCoordinateBallX <*> updatedCoordinateBallY <*> leapPosX
            
            // Event to determinate if the game is over 
            let condExitEvent : GameInfo Behavior Event = 
                (  // 'a Event
                    let predBeh = onceBehavior (conditionalGameOver <*> updatedCoordinateBallY)
                    
                    let ``a -> b`` = (fun () -> printf "Game Over"
                                                pureBehavior GameOver)
                    // initial Behavior "Game On"
                    // once the the Behavior condition becomes successful
                    // the event trigger the function which map
                    // Behvior a to Behavior b
                    fmap ``a -> b`` predBeh  )


            
            // executeBehaviorUntil ::'a Behavior -> 'a Behavior Event -> 'a Behavior
            executeBehaviorUntil gameInfo condExitEvent
        




        // This POC shows how continuous time lets you remove the concept of a frame 
        // and model the animation in a clean, natural and declarative way
        gameEngineState 0.0 0.0 0.0 leapPosXBehavior
    

















    let rendererGame state (gd : GraphicsDevice) = 
        match state with
        | GameOver -> LeapListenerSingleton.Instance.Value.Dispose()
        | State { xCoordBall = x; yCoordBall = y; xCoordLeap = xl } -> 
            drawPaddle xl paddleY (paddleSize /2.) gd
            drawBall x y ballRadius gd
    
    let startGame (game : Game) = 
        let stateB = gameEngine game
        (pureBehavior rendererGame) <*> stateB
