#light

namespace DynCol

 module Game = 
  open System
  open FsReactive.Misc
  open FsReactive.FsReactive
  open FsReactive.Integration
  open FsReactive.DynCol
  open FsReactive.Lib
  open Common.Random
  open Common.Vector
  open Xna.Main

  open Microsoft.Xna.Framework
  open Microsoft.Xna.Framework.Graphics
  open Microsoft.Xna.Framework.Input
   
  

  let rec mouseLeftButtonB = Beh (fun _ ->( Mouse.GetState().LeftButton.Equals(ButtonState.Pressed), fun () -> mouseLeftButtonB))
  
  let waitE dt = 
        let tmax = 2.0
        let bf t = let tmax = t+dt
                   let rec bf' t = if (t < tmax) then (None, fun () -> Evt bf')
                                                 else (Some (), fun () -> noneE)
                   (None, fun () -> Evt bf')
        Evt bf
           
         
// balls with collision

  let colf (x0,y0) (x1,y1) = (x1-x0)*(x1-x0)+(y1-y0)*(y1-y0) < 0.01

  let detectCol (id, x, y) l = 
    let rec proc l = 
        match l with
        |[] -> false
        |(id', x', y') :: t when id' <> id -> match colf (x,y) (x',y') with
                                                   |true -> true
                                                   |false -> proc t
        |_ :: t -> proc t
    proc l



  let mainGame (game:Game) = 
            let condxf = pureB (fun x -> x <= (-1.0) || x >= 1.0)
            let condyf = pureB (fun y -> y <= (-1.0) || y >= 1.0)
            let clickE = whenE mouseLeftButtonB
            
            let newBall t0 colB = 
                    let id = pureB (createId())
                    let speedChange x = let nx = -x * (1.0 + random()/2.0)
                                        if (Math.Abs nx) > 30.0 then 30.0 * ((float) (Math.Sign nx));
                                                                else nx
                    let rec x' = aliasB 0.0
                    let rec y' = aliasB 0.0
                    let rec condxE = whenE (condxf <.> (fst x')) --> speedChange
                    let rec condyE = whenE (condyf <.> (fst y')) --> speedChange
                    and speedx = stepAccumB (speedChange (randUnity()/3.0)) condxE          
                    and speedy = stepAccumB (speedChange (randUnity()/3.0)) condyE          
                    and x = bindAliasB (integrate speedx t0 0.0) x'
                    and y = bindAliasB (integrate speedy t0 0.0) y'
                    let ballB =  (tripleB() <.> id <.> x <.> y)
                    let hitE = (pureB detectCol) <.> ballB <.> colB 
                    untilB (someizeBf ballB) (whenE hitE --> noneB())
            let colB' = aliasB []
            let newBallE = (snapshotE clickE timeB) 
            let newBallE' = newBallE =>> ( fun (_, t0) -> [newBall t0 (fst colB')])
            let colB = bindAliasB (dyncolB [] newBallE') colB'
            colB


  
  let drawBall x y ballRadius (gd:GraphicsDevice)  = 
        let angles = Seq.toList (seq{for i in 1 .. 11 -> Math.PI*2.0*((float)i)/10.0})
        let pts = List.map (fun a-> (Vector.rot Vector.unit a) * ballRadius) angles
        let n_verts = List.length pts
        let random_vert _ =  Graphics.VertexPositionColor(Vector3(0.f, 0.f, 0.f), Graphics.Color.White)
        let vertex = Array.init n_verts random_vert
        let iter f      = List.iteri (fun i (Vector(x,y)) -> let (x', y') = f x y
                                                             vertex.[i].Position <- Vector3((float32)x', (float32)y', (float32) 0.0))        
                                     pts
        let f x' y' = x + x', y + y'          
        iter f 
        gd.DrawUserPrimitives(PrimitiveType.LineStrip, vertex, 0, n_verts-1)

  let renderer l (gd:GraphicsDevice) = 
    List.iter (fun (_, x, y) -> drawBall x y 0.05 gd) l


  let renderedGame (game:Game) = 
        let stateB = mainGame game
        (pureB renderer) <.> stateB 

  do use game = new XnaTest2(renderedGame)
     game.Run() 