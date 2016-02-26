namespace DynCol

module Game = 
    open System
    open FsFRPLib.Core
    open FsFRPLib

    open Common.Random
    open Common.Vector
    open Initializer
    open Microsoft.Xna.Framework
    open Microsoft.Xna.Framework.Graphics
    open Microsoft.Xna.Framework.Input
    
    // bool Behavior 
    let rec mouseLeftBtnBehavior = 
        Behavior(fun time -> (Mouse.GetState().LeftButton.Equals(ButtonState.Pressed), fun () -> mouseLeftBtnBehavior))
    
    // Game -> (int * float * float) list Behavior
    let gameEngine (game : Game) = 
        // (float -> bool) -> Behavior
        let predicateX = pureBeh (fun x -> x <= (-1.0) || x >= 1.0)
        // (float -> bool) -> Behavior
        let predicicateY = pureBeh (fun y -> y <= (-1.0) || y >= 1.0)
        // unit Event
        let clickMouseEvent = whenBehavior mouseLeftBtnBehavior
        
        // Time -> (int * float * float) list Behavior ->  (int * float * float) option Behavior
        let newBall time collisionBeh = 
            let id = pureBeh (createId())
            // float Behavior * (float -> unit)
            let x' = createBehavior 0.0
            // float Behavior * (float -> unit)
            let y' = createBehavior 0.0
            // (float -> float) Event
            // where (-->) :: 'a Event -> 'b -> 'b Event
            let predicateXEvent = whenBehavior (predicateX <*> (fst x')) --> incrSpeed
            // (float -> float) Event
            let predicateYEvent = whenBehavior (predicicateY <*> (fst y')) --> incrSpeed
            // float Behavior
            let speedX = accum (incrSpeed (randFloat() / 3.0)) predicateXEvent
            // float Behavior
            let speedY = accum (incrSpeed (randFloat() / 3.0)) predicateYEvent
            // float Behavior
            let x = bindBehaviors (integrate speedX time 0.0) x'
            // float Behavior
            let y = bindBehaviors (integrate speedY time 0.0) y'
            // (int * float * float) Behavior
            let ballBehavior = (pureBeh (fun x y z -> (x, y, z)) <*> id <*> x <*> y)
            // bool Behavior
            let collisionEvent = (pureBeh detectCol) <*> ballBehavior <*> collisionBeh
            // 'a Behavior -> 'a Behavior Event -> 'a Behavior 
            untillBehavior (someizeBf ballBehavior) (whenBehavior collisionEvent --> noneB())
        
        // (int * float * float) list Behavior * ((int * float * float) list -> unit)
        let colB' = createBehavior []
        // (unit * Time) Event
        let newBallE = (snapshotEvent clickMouseEvent timeBehavior)
        // (int * float * float) option Behavior List Event
        // where (==>) :: 'a Event -> ('a -> 'b) -> 'b Event
        let newBallE' = newBallE =>> (fun (_, t0) -> [ newBall t0 (fst colB') ])
        // (int * float * float) list Behavior
        let colB = bindBehaviors (dyncolB [] newBallE') colB'
        colB





    
    let drawBall x y ballRadius (gd : GraphicsDevice) = 
        let angles = 
            Seq.toList (seq { 
                            for i in 1..11 -> Math.PI * 2.0 * ((float) i) / 10.0
                        })
        
        let pts = List.map (fun a -> (Vector.rot Vector.unit a) * ballRadius) angles
        let n_verts = List.length pts
        let random_vert _ = Graphics.VertexPositionColor(Vector3(0.f, 0.f, 0.f), Graphics.Color.Red)
        let vertex = Array.init n_verts random_vert
        
        let iter f = 
            List.iteri (fun i (Vector(x, y)) -> 
                let (x', y') = f x y
                vertex.[i].Position <- Vector3((float32) x', (float32) y', (float32) 0.0)
                vertex.[i].Color <- Graphics.Color.Red) pts
        
        let f x' y' = x + x', y + y'
        iter f
        gd.DrawUserPrimitives(PrimitiveType.LineStrip, vertex, 0, n_verts - 1)
    
    let renderer l (gd : GraphicsDevice) = List.iter (fun (_, x, y) -> drawBall x y 0.05 gd) l
    
    let renderedGame (game : Game) = 
        let stateB = gameEngine game
        (pureBeh renderer) <*> stateB
    
    do use game = new GameInitializer(renderedGame)
       game.Run()
