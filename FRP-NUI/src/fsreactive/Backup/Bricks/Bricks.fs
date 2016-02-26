#light

namespace Bricks

  module Game =
           
    open FsReactive.Misc
    open FsReactive.FsReactive
    open FsReactive.Integration
    open FsReactive.DynCol
    open FsReactive.Lib
    open System
    open Common.Vector
    open Common.Random
    open Rendering
    open Xna.Main
    
    open Microsoft.Xna.Framework
    open Microsoft.Xna.Framework.Graphics
    open Microsoft.Xna.Framework.Input;
   

    let vectorNumClass = 
        {   plus = curry Vector.(+);
            minus = curry Vector.(-);
            mult = curry Vector.(*);
            div = curry Vector.(/);
            neg = Vector.neg
        }
    // some general purpose utility function    
    let (.*) a b = (pureB vectorNumClass.mult) <.> a <.> b 
    let (./) a b = (pureB vectorNumClass.div) <.> a <.> b 
    let (.+) a b = (pureB vectorNumClass.plus) <.> a <.> b 
    let (.-) a b = (pureB vectorNumClass.minus) <.> a <.> b 
    

    let rot90Clockwise (Vector (x,y)) = (Vector (y, -x))
    let rot90AntiClockwise (Vector (x,y)) = (Vector (-y, x))
       
    let cross (Vector (xm, ym)) (Vector (x1, y1)) (Vector (x2, y2)) =
        let sign (x:float) = Math.Sign x
        if not (sign y1 = sign y2)
        then    let t = -y1 /(y2 - y1)
                let xcross = x1+ t *(x2-x1)
                0.0 <= xcross && xcross <= xm
        else false

    let hitCorner corner pPrev pCur radius =
        let inCircle point centre radius = (Vector.length (point-centre)) <= radius
        if not (inCircle corner pPrev radius) && (inCircle corner pCur radius)
        then let dx = Vector.norm (pPrev - corner)
             let dy = rot90AntiClockwise dx
             let adaptVel v =   let r = dx * (-(Vector.dot dx v)) + dy * (Vector.dot v dy)
                                printf "corner = %A %A\n"  dx dy
                                r
               
             Some adaptVel
        else None
      
    let invY (Vector (x,y)) = Vector (x, -y)
    let invX (Vector (x,y)) = Vector (-x, y)

    let crossBox pll plr pul pur xp x radius =
        let hitpll = hitCorner pll xp x radius
        let hitplr = hitCorner plr xp x radius
        let hitpul = hitCorner pul xp x radius
        let hitpur = hitCorner pur xp x radius
        let crossBottom =   let xp' = xp + Vector(0.0, radius)
                            let x' = x + Vector(0.0, radius)
                            cross (plr - pll) (xp'-pll) (x'-pll)
        let crossUp = let xp' = xp - Vector(0.0, radius)
                      let x' = x - Vector(0.0, radius)
                      cross ((pur - pul)) (invY (xp'-pul)) (invY (x'-pul))
        let crossRight = let xp' = xp - Vector(radius, 0.0)
                         let x' = x - Vector(radius, 0.0)
                         cross (rot90Clockwise (pur-plr)) ((rot90Clockwise (xp'-plr))) ((rot90Clockwise (x'-plr)))              
        let crossLeft = let xp' = xp + Vector(radius, 0.0)
                        let x' = x + Vector(radius, 0.0)
                        cross (rot90AntiClockwise (pur-plr) |> invX) (rot90AntiClockwise (xp'-pll) |> invX) (rot90AntiClockwise (x'-pll) |> invX)    
        let choice b v = if b then Some v else None
        let preds = [   hitpll;
                        hitplr;
                        hitpul;
                        hitpur;
                        choice crossBottom ( fun (Vector (x, y)) -> Vector (x, -y));
                        choice crossUp (fun (Vector (x, y)) -> Vector (x, -y));
                        choice crossRight (fun (Vector (x, y)) -> Vector (-x, y));
                        choice crossLeft (fun (Vector (x, y)) -> Vector (-x, y))
                    ]
        let pred = List.filter (fun x->isSome x)  preds
        match pred with
        |[] -> None
        |h::t -> h


    let brickWidth = 0.2
    let brickHeight = 0.1

    let paddleY = -0.95;
    let paddleHalfLength = 0.1;   

    let bricksCoord = 
        let xs = List.init 10 (fun i -> -1.0 + (float) i * brickWidth)
        let ys = List.init 5 (fun i -> 0.9 - (float) i * brickHeight)
        List.map (fun x -> List.map (fun y -> (x, y)) ys) xs |> List.concat

    type Brick = Brick of (int * Vector) // id, x, y

    let mkHits ballRadius bricks pBallOption ballOption  = 
        match pBallOption, ballOption with
        |Some pBall, Some ball ->
            let proc brick = let pll = brick 
                             let plr = brick + Vector(brickWidth, 0.0)
                             let pul = brick + Vector(0.0, brickHeight) 
                             let pur = brick + Vector(brickWidth, brickHeight) 
                             crossBox pll plr pul pur pBall ball ballRadius
            let r = List.map (fun (Brick (id, pos)) ->  match proc pos with
                                                        |Some f -> (Some (id, f))
                                                        |None -> None) bricks
            let r' = catOption r
            r'
            |_ -> []
       
    let mkBrick pos hitsB = 
        let id = createId()
        let isHit hits = let r = List.exists (fun (id', _) -> id = id') hits
                         r 
        let isHitE = whenE ((pureB isHit) <.> (hitsB)) --> (noneB())
        let brickB = untilB  (pureB (Some (Brick (id, pos)))) isHitE
        brickB

    let mkBricks hitsB = 
        let brickBs = List.map (fun pos -> mkBrick (Vector pos) hitsB) bricksCoord
        dyncolB brickBs noneE
         
    let mkVelocity v0 hitsB ballB xPaddleB ballRadius = 
        let applyB = (pureB (fun f b -> f b))
        let hitWallx (Vector (x, y))  = 
            match (x <= -1.0 || x>= 1.0) with
            |(false) -> None
            |(true) -> Some (fun v -> (Vector.rot v (randRange -0.01 0.01)) |> invX)
        let hitWally (Vector (x, y)) xPaddle= 
            match ((y-ballRadius <= paddleY && xPaddle-paddleHalfLength <= x && x <= xPaddle+paddleHalfLength) || y>= 1.0) with
            |(false) -> None
            |(true) -> Some (fun v -> (Vector.rot v (randRange -0.01 0.01)) |> invY)
        let hitBricks hits =
            List.fold (fun acc (_, f) ->
                            match acc with
                            |None -> Some f
                            |Some acc -> Some (acc >> f))None hits
        let hitWallxE = whenBehaviorE ((pureB hitWallx) <.> ballB)
        let hitWallyE = whenBehaviorE ((pureB hitWally) <.> ballB <.> xPaddleB)
        let hitBrickE = whenBehaviorE ((pureB hitBricks) <.> hitsB)
        let comp a b  = a >> b
        let hitE =  orE comp (orE comp hitWallxE hitWallyE) hitBrickE
        let vB = stepAccumB v0 hitE
        let adjust (Vector (x, y)) = let y' =   if  Math.Abs(y) <= 0.2 
                                                then 0.2 * ((float) (Math.Sign(y)))
                                                else y
                                     Vector( x,y')       
        ((pureB adjust) <.> vB)  

     
    let mkPaddle x0 (game:Game) =
        let mousePos = mousePos game
        let rec mousePosEvt = Evt (fun _ -> (mousePos(), fun() -> mousePosEvt))
        let mousePosB  = stepB (0.0, 0.0) mousePosEvt
        let mousePosXB =  pureB  (fun x -> if x - paddleHalfLength < -1.0 
                                           then  (-1.0 )
                                           else if x - paddleHalfLength > 1.0
                                                then 1.0 
                                                else x)
                              <.> (pureB fst <.> mousePosB)
        mousePosXB

    type State = 
        { 
            ball:Vector option
            bricks:Brick list
            xpaddle : float
            nbBalls:int
        }

    let rec keyboardInputG key = Beh (fun _ -> ( Keyboard.GetState().IsKeyDown(key), fun () -> keyboardInputG key))
    let rec startCommandB =  (keyboardInputG Keys.Enter .||.  keyboardInputG Keys.Space)

    let ballRadius = 0.05

    let rec mkBall xPaddleB hitsB ballRadius x0 t0  = 
                let integrate = integrateGenB vectorNumClass
                let xB' = aliasB x0
                let v0 = (Vector.rot Vector.unit (Math.PI/4.0)) * 1.5
                let velB = mkVelocity v0 hitsB (fst xB') xPaddleB ballRadius
                let xB = bindAliasB (integrate velB t0 x0 ) xB' 
                let xpB = delayB xB ( x0)
                let ballOutE = whenE ((pureB (fun (Vector(x, y)) -> y <= -1.0  )) <.> xB)
                let ballB = (coupleB()) <.>  (someizeBf xB) <.> (someizeBf xpB)  //|> tronB "balls option"  
                           
                let ballB' = untilB (ballB)
                                    (ballOutE --> untilB ((coupleB()) <.> (noneB()) <.> (noneB())) 
                                                         (waitE 2.0 =>> ( fun () -> startB ( mkBall xPaddleB hitsB ballRadius x0))))
                ballB' 

    let render {ball=ballOption
                bricks=bricks
                xpaddle = xpaddle} (gd:GraphicsDevice) = 
        let drawBrick'  (Brick (_, (Vector(x, y)))) = drawBrick x y   
        let drawBall' (Vector(x, y)) = drawBall x y ballRadius 
        let drawPaddle' x paddleY paddleHalfLength = drawPaddle x paddleY paddleHalfLength 
        List.iter (fun brick -> drawBrick' brick gd )  bricks
        match ballOption with
        |Some ball -> drawBall' ball gd 
        |None -> ()
        drawPaddle' xpaddle paddleY paddleHalfLength gd 

    let rec game (theGgame:Game) = 
        let noGame =  {   ball=Some (Vector( 0.0, 0.0))
                          bricks=[]
                          xpaddle = 0.0
                          nbBalls = 3}
        
        let rec startGame t0 = 
            let x0 = (Vector.zero)

            let xPaddleB = mkPaddle 0.0 theGgame
            let hitsB' = aliasB []
            let ballB =  mkBall xPaddleB (fst hitsB') ballRadius x0 t0 |> memoB
            let xB =  (pureB fst) <.> ballB   |>memoB      //|> tronB ""
            let xpB = (pureB snd) <.> ballB    

            let bricksB' = aliasB []
            let hitsB = bindAliasB ((pureB (mkHits ballRadius)) <.> (fst bricksB') <.> xpB <.> xB )  hitsB'
            let bricksB =  bindAliasB (mkBricks hitsB) bricksB'
            let gameB = (pureB (fun ball bricks xpaddle -> 
                                    {   ball=ball
                                        bricks=bricks
                                        xpaddle = xpaddle
                                        nbBalls = 0 })) <.> xB <.> bricksB <.> xPaddleB
            gameB
        let rec proc () = 
            let  mkGame t0 = let stateB = (startGame t0) 
                             stateB
            untilB (pureB noGame)
                   (whenE (startCommandB) --> (startB mkGame))
        let stateB = proc()  |> memoB
        let decrBallNbE = whenE ((pureB (fun state -> state.ball=None)) <.> stateB) --> (fun x -> x-1) 
        let nbBallsB = stepAccumB 3 decrBallNbE  // |> tronB "balls " 
        let stateB' = (pureB render) <.> stateB
        let stateB'' = untilB stateB' (whenE (nbBallsB .<=. (pureB 0)) =>> fun () -> game(theGgame))
        stateB''

        

    do use game = new XnaTest2(game)
       game.Run()

