#light


namespace Asteroids

  module Game =
           
    open FsReactive.Misc
    open FsReactive.FsReactive
    open FsReactive.Integration
    open FsReactive.DynCol
    open FsReactive.Lib
    open System
    open Common.Vector
    open Common.Random
    open Asteroids.Data
    open Asteroids.Rendering
    open Xna.Main
    
    open Microsoft.Xna.Framework
    open Microsoft.Xna.Framework.Graphics
    open Microsoft.Xna.Framework.Input;

    
    // vector num class required by integration

    let vectorNumClass = 
        {   plus = curry Vector.(+);
            minus = curry Vector.(-);
            mult = curry Vector.(*);
            div = curry Vector.(/);
            neg = Vector.neg
        }

    
    // return true is the point (x,y) is within a (-1, 1) box
    let inBoxPred (Vector (x,y)) = 
        let proc x = -1.0 <= x && x <= 1.0 
        if  (proc x && proc y) 
        then  true
        else  //printf "out\n"
              false   
    let adaptToBox (Vector (x,y)) = 
        let rec proc x = if x < -1.0 then proc (x+2.0) else if 1.0 < x then proc (x-2.0) else x 
        Vector (proc x, proc y) 

    // some general purpose utility function    
    let (.*) a b = (pureB vectorNumClass.mult) <.> a <.> b 
    let (./) a b = (pureB vectorNumClass.div) <.> a <.> b 
    let (.+) a b = (pureB vectorNumClass.plus) <.> a <.> b 
    let (.-) a b = (pureB vectorNumClass.minus) <.> a <.> b 
    
    // movement

    let mkMovement t0 x0 velocityB =
        let integrate = integrateGenB vectorNumClass
        let rec proc t0 x0 e0 = 
            let x0' = e0 t0 x0
            let x = integrate velocityB t0 x0'
            let boxE = (whileE ((pureB (inBoxPred >> not)) <.> x)) --> (fun _ x -> adaptToBox x)
            Disc (x, boxE, proc)
        discontinuityE (proc t0 x0 (fun _ x -> adaptToBox x))

    // description of game objets

    // some constants
    let nominalMeteorVelocity = 0.3
    let bulletVelocity = 1.0
    let bulletLifeTime = 2.0
    let shipFriction = 1.0
    let newShipCreationDelay = 2.0

        
    let rec mouseLeftButtonB = Behavior (fun _ ->( Mouse.GetState().LeftButton.Equals(ButtonState.Pressed), fun () -> mouseLeftButtonB))
    let rec mouseRightButtonB = Behavior (fun _ -> ( Mouse.GetState().RightButton.Equals(ButtonState.Pressed), fun () -> mouseRightButtonB))
    let rec mouseMiddleButtonB = Behavior (fun _ -> ( Mouse.GetState().MiddleButton.Equals(ButtonState.Pressed), fun () -> mouseMiddleButtonB))
    let rec keyboardInputG key = Behavior (fun _ -> ( Keyboard.GetState().IsKeyDown(key), fun () -> keyboardInputG key))
    let rec fireCommandB =  (keyboardInputG Keys.Enter .||.  keyboardInputG Keys.O .||. keyboardInputG Keys.Z)
    let rec thrustCommandB = keyboardInputG Keys.Space .||. keyboardInputG Keys.Up
    let rec leftCommandB =  (keyboardInputG Keys.Left .||. keyboardInputG Keys.A .||. mouseLeftButtonB)
    let rec rightCommandB = (keyboardInputG Keys.Right .||. keyboardInputG Keys.P.||. mouseRightButtonB)
    let rec shieldCommandB = keyboardInputG Keys.Z

    // jet engine 
    let rec jetB = 
        let e = (loopE (whenE (periodicB 0.025)) [JetSmall; JetMedium; JetBig]) =>> snd
        let b = stepB JetSmall e
        (pureB (fun thrust jet -> if thrust then Some jet else None)) <.> thrustCommandB <.> b

    // ship angle 
    let mkVelocityAngle rightTurnB leftTurnB t0 (angle:float) increment = 
        let proc b = if b then increment else 0.0
        let angleVelocity = (((pureB proc) <.> leftTurnB) .-. ((pureB proc) <.> rightTurnB))
        memoB <| (integrate angleVelocity t0 angle)
        
    // shield command
    let shieldDuration = 1.0
    let shieldReloadDelay = 2.0

    let shieldB = 
        let shieldCommandE = (whenE shieldCommandB )
        let rec proc() = memoB (untilB falseB  (shieldCommandE =>> (fun () -> untilB trueB (waitE shieldDuration =>> (fun() -> untilB falseB (waitE shieldReloadDelay =>> proc))))))
        proc()

    // create a meteor of size=msize with initial position = x0 at initial time = t0
    // and a nominal velocity of v0
    // the meteor change velocity when it hits the shipB shield
    let mkMeteor t0 x0 (nominalMeteorVelocity:Vector) msize shipB shieldB = 
     
        let meteorPos' = aliasB x0
        let speedFactor =  MeteorSize.speedFactor msize
        let meteorVelocity = nominalMeteorVelocity * speedFactor
        let shieldContact ship meteorPos shield = 
            match ship with
            |Some (Ship (_, shipPos, _, _)) -> shield && Vector.length (meteorPos - shipPos) < 0.2 
            |None -> false
        let rec shieldContactE = (whenE ((pureB shieldContact) <.> shipB <.> (fst meteorPos') <.> shieldB)) --> (fun x -> Vector.neg x)
        let meteorVelocityB = stepAccumB meteorVelocity shieldContactE |> memoB
        
        let meteorPos = bindAliasB (mkMovement t0 x0 meteorVelocityB) meteorPos'
        let meteorB =   let id = (createId())
                        (pureB (fun x -> Meteor (id, x, msize))) <.> meteorPos
        meteorB

    
    // create a behavior holding a list of hits between meteors and bullets 
    let mkMeteorBulletHit meteorsB bulletsB = 
        let colPred (Meteor (_, meteorPos, msize)) (Bullet (_, bulletPos)) = 
            Vector.length (meteorPos - bulletPos) <= (MeteorSize.size msize)
        let detect meteors bullets = 
            let rec proc lml b rml = 
                match rml with
                |[] -> None
                |h::t -> if colPred h b
                         then Some ((lml @ t), (h, b))
                         else proc (lml @ [h]) b t
            let proc' state b = 
                let (hits, ml) = state
                match proc [] b ml with
                |Some (nml, hit) -> (hit::hits, nml) 
                |None -> state             
            let hits = List.fold proc' ([], meteors) bullets |> fst 
            if not (List.isEmpty hits) 
            then printf "hit\n"
            else ()
            hits
        memoB <| (pureB detect) <.> meteorsB <.> bulletsB

        
    // create a meteor that can be destroyed 
    let rec mkBreakableMeteor meteorB hitsB =
        let hitPred (Meteor (id, _, _)) hits = List.exists (fun ((Meteor (idb, _, _)), _) -> let r = id = idb
                                                                                             r) hits
        let hitE = whenE ((pureB hitPred) <.> meteorB <.> hitsB) --> (noneB())
        let breakableMeteorB =  untilB (someizeBf meteorB) hitE
        breakableMeteorB

        
    // create a a Meteor option representing the meteor that hit the ship
    let mkShipMeteorHit shipB meteorsB = 
        let hitPred (Ship (_, shipPos, _, _)) (Meteor (_, meteorPos, msize)) = 
            Vector.length (shipPos - meteorPos) <= (MeteorSize.size msize + 3.0/40.0)
        let detect ship meteors = 
            match ship with
            |Some ship -> List.tryFind (hitPred ship) meteors
            |None -> None
        (pureB detect) <.> shipB <.> meteorsB
  

    // create a bullet option with the initial position = x0 at time = t0 with the velocity = bulletVelocityB.
    // the bullet lifetime ends after a given time or when it hits a meteor
    let mkBullet t0 x0 bulletVelocityB hitsB =
        let bulletPos = mkMovement t0 x0 bulletVelocityB 
        let bulletB =   let id = (createId())
                        (pureB (fun pos -> Bullet (id, pos))) <.> bulletPos
        let ageE = (whenE (timeB .-. (pureB t0) .>. (pureB bulletLifeTime)) --> (noneB()))
        let hitPred (Bullet (id, _)) hits = List.exists (fun (_,  (Bullet (idb, _))) -> id = idb) hits
        let hitE = whenE ((pureB hitPred) <.> bulletB <.> hitsB) --> (noneB())
        let bulletB' =  memoB (untilB (someizeBf bulletB) (ageE .|. hitE))
        bulletB'
        


    // make a new ship     
    let rec mkShipDynamics t0 angleB thrustB nbShipsB hitB =
        let integrate = integrateGenB vectorNumClass
        let (.* ) (a:Behavior<Vector>) b = (pureB (*)) <.> a <.> b 
        let (.- ) (a:Behavior<Vector>) b = (pureB (-)) <.> a <.> b 
        
        let shipVelocityB' = aliasB (Vector.zero)
        let accB =  ((pureB Vector.rot) <.> thrustB <.> angleB) .-  ((pureB (fun v -> v * shipFriction)) <.> (fst shipVelocityB')) 
        let shipVelocityB = bindAliasB (integrate accB t0  (Vector.zero)) shipVelocityB'
        
        let shipPositionB' = aliasB (Vector.zero)
        let shipPositionB = bindAliasB (mkMovement t0 (Vector.zero) shipVelocityB) shipPositionB'
        
        let shipB' = aliasB (None)
        let shipToCreateE = whenE ((pureB (fun nbShips ship -> nbShips > 0 && not (isSome ship))) <.> nbShipsB <.> (fst shipB') )
        let shipToCreateE' =  shipToCreateE =>> 
                              (fun () -> startB (fun t0 ->  let b = mkShipDynamics t0 angleB thrustB nbShipsB hitB
                                                            untilB (noneB()) (waitE newShipCreationDelay --> b )))
        let destroyedShipE = whenE ((pureB (fun hit -> isSome hit)) <.> hitB) --> (untilB (noneB()) shipToCreateE')
        let shipB = let id = (createId())
                    bindAliasB (untilB (someizeBf ((pureB (fun shipPos angle jet -> Ship (id, shipPos, angle, jet))) <.> shipPositionB <.> angleB <.> jetB)) destroyedShipE) shipB'
        memoB shipB

        
       
    // fire bullet event
    // firingCommandB : bool event : true when firing button is pressed
    let mkFire shipB hitsB =
        let fireE = whenE fireCommandB 
        let proc (ship, t0) = 
             match ship with
             |Some (Ship (_, x, angle, _)) -> 
                                        let v = Vector.rot (Vector.unit) angle * bulletVelocity
                                        let bulletPos = x +  (Vector.rot (Vector.unit) angle) * 0.1
                                        let r = ((mkBullet t0 bulletPos (pureB v) hitsB))
                                        [r]
             |None -> []
        let newBulletE = snapshotBehaviorOnlyE fireE (coupleB() <.> (shipB) <.> timeB) =>> proc
        newBulletE
        

    // create meteor generator
    let rec mkMeteorGenerator meteorsB shipB shieldB thereAreMeteorsE nbNewMeteorsB hitsB = 
      let rec splitMeteorProc shipB  =  
            let proc t0 (Meteor ( _, x, msize)) = 
                                             if msize = MeteorSize.Small 
                                             then   []
                                             else   let msize' = MeteorSize.smaller msize
                                                    let v1 = Vector.rot Vector.unit (randAngle()) * nominalMeteorVelocity
                                                    let v2 = Vector.rot Vector.unit (randAngle()) * nominalMeteorVelocity
                                                    let meteor1B = mkMeteor t0 x v1 msize' shipB shieldB
                                                    let r1 = (mkBreakableMeteor meteor1B hitsB)
                                                    let meteor2B = mkMeteor t0 x v2 msize' shipB shieldB
                                                    let r2 = (mkBreakableMeteor meteor2B hitsB)
                                                    ([r1; r2])
            // meteor split by a bullet hit
            let splitMeteors hits t0 = 
                    List.fold (fun state (meteor,_) -> proc t0 meteor @ state) [] hits
            let splitMeteorsB =  (((pureB splitMeteors) <.> hitsB <.> timeB))
            // new set of meteors when all have been destroyed
            let newSetOfMeteors nms ms nbMeteors = 
                    if List.isEmpty nms
                    then if List.isEmpty ms
                         then let meteors = List.map (fun _ -> startB (fun t0 -> let meteorB = mkMeteor t0 (randVector()) ((Vector.rot Vector.unit (randAngle())) * nominalMeteorVelocity) MeteorSize.Big shipB shieldB
                                                                                 mkBreakableMeteor meteorB hitsB))
                                                     (Seq.toList ( seq {for i in 1 .. nbMeteors -> ()}))
                              meteors
                         else []
                    else []

            let newSetOfMeteorsB = (pureB newSetOfMeteors) <.> splitMeteorsB <.> meteorsB <.> nbNewMeteorsB
            let newSetOfMeteorsB' = untilB (pureB []) (waitE 2.0 --> newSetOfMeteorsB)
            let newSetOfMeteorsE = whenE ((pureB (fun l ->  (List.isEmpty l))) <.> meteorsB)
            let newMeteorsB = untilB splitMeteorsB 
                                     (newSetOfMeteorsE --> newSetOfMeteorsB')
            
            let newMeteorsB' = untilB newMeteorsB 
                                      (thereAreMeteorsE =>> fun () -> splitMeteorProc shipB)
            newMeteorsB'
      splitMeteorProc shipB

    // the incrementing score
    let mkScore hitsB = 
        let sumScore acc ((Meteor (_, _, size)),_) = acc + (MeteorSize.score size)
        let incrScoreB = (pureB (fun hits -> let r = List.fold sumScore 0 hits
                                             match r with
                                             |0 -> None
                                             |incr -> Some (fun x -> x + incr))) <.> hitsB
        let scoreB = stepAccumB 0 (whenBehaviorE incrScoreB)
        scoreB


    // nb of bonus ship
    let mkBonusShips scoreB  = 
        let minScoreB' = aliasB 2000
        let incrMinScoreE = whenE ((pureB (fun score minScore -> score >= minScore)) <.> scoreB <.> (fst minScoreB'))
        let minScoreB = bindAliasB (stepAccumB 2000 (incrMinScoreE --> (fun x -> x+2000))) minScoreB'
        (pureB (fun s -> s/2000-1)) <.> minScoreB

    // scale used during a ship explosion
    
    // create an exploding ship
    let rec mkDestroyedShip shipB hitB  = 
        let scaleB() = stepB 0.1 (iterE (whenE (periodicB 0.01)) [ 0.2; 0.3; 0.4; 0.7; 0.9; 1.1; 1.4; 1.7; 2.0; 0.0] =>> snd)
        let proc shipOption =
            match shipOption with
            |Some (Ship (_, x, angle, _)) -> 
                                let scaleB' = startB (fun _ -> scaleB())
                                let nsB = pureB (fun scale -> Some (DestroyedShip (x, angle, scale))) <.> scaleB'
                                untilB nsB (whenE ((pureB (fun scale -> scale = 0.0)) <.> scaleB') =>> (fun _ -> mkDestroyedShip shipB hitB))
            |None -> noneB()

        let shipExplosionE = (snapshotBehaviorOnlyE (whenE ((pureB (fun hit -> isSome hit)) <.> hitB)) shipB) =>> proc
        untilB (noneB()) shipExplosionE

    type GameState = 
        {   nbShips: int
            ship:Ship option
            meteors:Meteor list
            bullets:Bullet list
            destroyedShip : DestroyedShip option
            shieldOn : bool
            score:int
        }

    let mainGame (game:Game) = 
        let standbyGame = {nbShips = 3; ship = None; meteors = []; bullets = []; score = 0; destroyedShip = None; shieldOn=false}
        let rec startGame t0 = 
            let angleB =  mkVelocityAngle (rightCommandB ) ( leftCommandB) 0.0 1.0 4.0
            let thrustB = (pureB (fun b -> if b then (Vector (1.25, 0.0)) else (Vector (0.0, 0.0)))) <.> thrustCommandB

            let shipB' = aliasB None
            let hitsB' = aliasB []
            let hitB' = aliasB None

            // detect ship creation
            let createShipE = whenE ((pureB (fun ship -> isSome ship)) <.> (fst shipB'))

            let nbShipsB = stepAccumB 3 (createShipE --> (fun n -> n-1))
            let scoreB = mkScore (fst hitsB')
            let bonusShip = mkBonusShips scoreB
            let nbTotalShipsB = (pureB (+)) <.> nbShipsB <.> bonusShip

            let shipB = bindAliasB (mkShipDynamics 0.0 angleB thrustB nbTotalShipsB  (fst hitB')) shipB'

            let meteorsB' = aliasB ([])

            let mkMeteors = List.map (fun _ -> let meteorB = mkMeteor t0 (randVector()) ((Vector.rot Vector.unit (randAngle())) * nominalMeteorVelocity) MeteorSize.Big shipB shieldB
                                               mkBreakableMeteor meteorB (fst hitsB'))
                                      [] //[1;2;3;4]
            let thereAreMeteorsE = whenE ((pureB (fun l -> not (List.isEmpty l))) <.> (fst meteorsB'))
            let nbNewMeteorsB  = 
                    let newMeteorsE = ((iterE thereAreMeteorsE [2;3;4;5;6;7;8;9;10;11]) =>> (fun (_, n) -> n))
                    (memoB (stepB 1 newMeteorsE))

            let meteorCreatorE = snapshotBehaviorOnlyE (someE ())  (mkMeteorGenerator (fst meteorsB') shipB shieldB thereAreMeteorsE nbNewMeteorsB (fst hitsB'))
            let meteorsB = bindAliasB (dyncolB  mkMeteors meteorCreatorE)  meteorsB'
            let delayedMeteorsB =  untilB (pureB []) (waitE 2.0 --> ( meteorsB))

            let bulletsB = (dyncolB [] (mkFire shipB  (fst hitsB'))) 
        
            let hitsB = bindAliasB (mkMeteorBulletHit meteorsB bulletsB) hitsB'
            let hitB = bindAliasB (mkShipMeteorHit shipB meteorsB) hitB'

            let destroyedShipB = mkDestroyedShip shipB hitB

            let stateB =  memoB ((pureB (fun nbShips ship meteors bullets score destroyedShip shieldOn -> {nbShips = nbShips; ship = ship; meteors = meteors; bullets = bullets; score = score; destroyedShip = destroyedShip; shieldOn= shieldOn}))
                                                <.> nbTotalShipsB
                                                <.> shipB 
                                                <.> delayedMeteorsB
                                                <.> bulletsB
                                                <.> scoreB
                                                <.> destroyedShipB
                                                <.> shieldB)
            let endGameE = whenE ((pureB (fun m -> m.nbShips = 0 && not (isSome m.ship) && not (isSome m.destroyedShip))) <.> stateB)
            let newGameE = (whenE fireCommandB) =>> (fun () -> startGame 0.0)
            let stateB' = untilB stateB (endGameE --> untilB (pureB standbyGame) newGameE) |> memoB
            seqB hitB 
                 <| (seqB hitsB 
                          <| seqB nbNewMeteorsB stateB')
            
            
        untilB (pureB standbyGame) ((whenE fireCommandB) =>> (fun() -> startGame 0.0))

    let renderer {  nbShips = nbShips
                    ship=ship
                    meteors=meteors 
                    bullets=bullets; score=score
                    destroyedShip = destroyedShip
                    shieldOn = shieldOn}
                 (gd:GraphicsDevice) = 
        match ship with
        |Some ship' -> drawShip' gd ship' 
        |None -> ()
        match destroyedShip with
        |Some ship' ->  drawExplodingShip' gd ship' 
        |None -> ()
        List.iter (fun e -> drawMeteor' gd e) meteors
        List.iter (fun e -> drawBullet' gd e) bullets
        drawRemainingShips gd nbShips
        drawShield' gd shieldOn ship


    let renderedGame (game:Game) = 
        let stateB = mainGame game
        (pureB renderer) <.> stateB 

    do use game = new XnaTest2(renderedGame)
       game.Run()