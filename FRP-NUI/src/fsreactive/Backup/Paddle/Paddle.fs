#light



namespace Paddle

 module Game =
  open System
  open FsReactive.Misc
  open FsReactive.FsReactive
  open FsReactive.Integration
  open FsReactive.Lib
  open Paddle.Rendering
  open Common.Random
  open Xna.Main

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
           
            let mousePos = mousePos game
            let rec mousePosEvt = Evt (fun _ -> (mousePos(), fun() -> mousePosEvt))
            let mousePosB  = stepB (0.0, 0.0) mousePosEvt
            let mousePosXB =  pureB  (fun x -> if x - paddleHalfLength < -1.0 
                                               then  (-1.0 )
                                               else if x - paddleHalfLength > 1.0
                                                    then 1.0 
                                                    else x)
                              <.> (pureB fst <.> mousePosB)
            let rec sys t0 xball0 yball0 xpad0 mousePosXB = 
                let xpad = mousePosXB
                let xball' = aliasB xball0
                let yball' = aliasB yball0
                let rec condxE = (whenE (condxf <.> (fst xball'))) --> (fun x -> -x)
                let vballx = stepAccumB (0.3) condxE
                let rec condyE = (whenE (condyf <.> (fst xball') <.> (fst yball') <.> xpad)) --> (fun x -> -x)
                let vbally = stepAccumB (-0.45) condyE

                let xball = memoB ( bindAliasB (integrate vballx t0   xball0) xball' )
                let yball = memoB ( bindAliasB (integrate vbally  t0 yball0) yball' )
                let state = (pureB (fun xb yb xp -> State {xball = xb ; yball = yb ; xpaddle = xp}))
                             <.> xball <.> yball <.> xpad
                let condeExitE =  (whenE (condExitYf <.> yball)) =>> (fun _ -> printf "stop\n" 
                                                                               pureB End)
                untilB state condeExitE
            sys 0.0 0.0 0.0 0.0 mousePosXB

  let renderer state (gd:GraphicsDevice) = 
         match state with
         |End -> ()
         |State  {xball=x; yball=y ; xpaddle = xp} as s->
            drawPaddle xp paddleY paddleHalfLength gd 
            drawBall x y ballRadius gd 
   
  let renderedGame (game:Game) = 
        let stateB = mainGame game
        (pureB renderer) <.> stateB 

  do use game = new XnaTest2(renderedGame)
     game.Run() 











