namespace PongGame

 module Game =
  open System
  open FsFRPx.FsFRPx
  open FsFRPx.Helpers
  open FsFRPx
  open PongGame.Rendering
  open PongGame.PongGameInitializer

  open Microsoft.Xna.Framework
  open Microsoft.Xna.Framework.Graphics
  open Microsoft.Xna.Framework.Input

  let paddleHalfLength = 0.1
  let paddleY = -0.95
  let ballRadius = 0.05



  type StateRec = { xball : float; yball : float ; xpaddle : float }
    type State = 
        |State of StateRec
        |End

  let mainGame (game:Game) = 

            let condxf = pureB (fun x -> x-ballRadius <= (-1.0) || x+ballRadius >= 1.0)
            let condyf = pureB (fun x y xpad -> y+ballRadius >= 1.0 || ((xpad - paddleHalfLength) <= x 
                                                && x <= (xpad + paddleHalfLength)
                                                &&  y <= (paddleY+ballRadius)))
            let condyf' = pureB (fun x y xpad -> y+ballRadius >= 1.0 ||  y-ballRadius <= (-1.0))
            let condExitYf = pureB (fun y ->  y <= (-1.0))
           
            let leapPos = leapPos game
            let rec leapPosEvt = Event (fun _ -> (leapPos(), fun() -> leapPosEvt))
            let leapPosB  = stepB (0.0, 0.0) leapPosEvt
            let leapPosXB =  pureB  (fun x -> if x - paddleHalfLength < -1.0 
                                               then  (-1.0 )
                                               else if x - paddleHalfLength > 1.0
                                                    then 1.0 
                                                    else x)
                              <.> (pureB fst <.> leapPosB)
            let rec sys t0 xball0 yball0 xpad0 leapPosXB = 
                let xpad = leapPosXB
                let xball' = aliasB xball0
                let yball' = aliasB yball0
                let rec condxE = (whenE (condxf <.> (fst xball'))) --> (fun x -> -x)
                let vballx = stepAccumB (0.4) condxE
                let rec condyE = (whenE (condyf <.> (fst xball') <.> (fst yball') <.> xpad)) --> (fun x -> -x)
                let vbally = stepAccumB (-0.45) condyE

                let xball = memoB ( bindAliasB (integrate vballx t0   xball0) xball' )
                let yball = memoB ( bindAliasB (integrate vbally  t0 yball0) yball' )
                let state = (pureB (fun xb yb xp -> State {xball = xb ; yball = yb ; xpaddle = xp}))
                             <.> xball <.> yball <.> xpad
                let condeExitE =  (whenE (condExitYf <.> yball)) =>> (fun _ -> printf "Game Over\n" 
                                                                               pureB End)
                untilB state condeExitE
            sys 0.0 0.0 0.0 0.0 leapPosXB

  let renderer state (gd:GraphicsDevice) = 
         match state with
         |End -> LeapListenerSingleton.Instance.Value.Dispose()
         |State  {xball=x; yball=y ; xpaddle = xp} as s->
            drawPaddle xp paddleY paddleHalfLength gd 
            drawBall x y ballRadius gd 
   
  let renderedGame (game:Game) = 
        let stateB = mainGame game
        (pureB renderer) <.> stateB 




  do use game = new GameInitializer(renderedGame)
     game.Run() 







     



