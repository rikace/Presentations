#light

namespace Rectangles

 module Main = 
  open System
  open FsReactive.Misc
  open FsReactive.FsReactive
  open FsReactive.Integration
  open FsReactive.Lib
  open Common.Vector
  open Xna.Main

  open Microsoft.Xna.Framework
  open Microsoft.Xna.Framework.Graphics
  open Microsoft.Xna.Framework.Input
  

  let (.*) a b = (pureB (*)) <.> a <.> b 
  let (.-) a b = (pureB (-)) <.> a <.> b 
  let (.+) a b = (pureB (+)) <.> a <.> b 
  let cosB = pureB Math.Cos
  let sinB = pureB Math.Sin
  let (.<=) a b = pureB (<=) <.> a <.> b
  let (.>=) a b = pureB (>=) <.> a <.> b
  
  type Rect = Rect of (Vector * Vector)

  type State =  { 
    rectangles : Rect list
    currentRec : Rect option
    }

        
  let rec mouseLeftButtonB =  Beh (fun _ ->(Mouse.GetState().LeftButton.Equals(ButtonState.Pressed), fun () -> mouseLeftButtonB))
                            
  let rec mouseRightButtonB = Beh (fun _ -> ( Mouse.GetState().RightButton.Equals(ButtonState.Pressed), fun () -> mouseRightButtonB))
                             

  let mainGame (game:Game) = 

        let mousePos = mousePos game
        let rec mousePosEvt = Evt (fun _ -> (mousePos(), fun() -> mousePosEvt))
        let mousePosB  = stepB (0.0, 0.0) mousePosEvt |> memoB
        let leftClickE = whenE mouseLeftButtonB |> memoE
        let rightClickE = whenE mouseRightButtonB |> memoE
       
        let rec rectB () = 
                untilB (noneB()) (snapshotBehaviorOnlyE (leftClickE)  mousePosB =>> (fun (x, y) -> mkRect x y))
        and mkRect x y =
                let movingRecB = (pureB  (fun (x, y) -> Rect (Vector (-x, -y), Vector (x, y))))
                                 <.> mousePosB
                untilB (someizeBf movingRecB) ((leftClickE .|. (rightClickE)) =>> (fun _ -> rectB ()))
        let rectB' = rectB ()
        let stepProc rect rects = 
            match rect with
            |Some rect -> rect :: rects
            |None -> rects
        let newRectE = snapshotE    (rightClickE)  rectB' =>> (fun (_, rect) -> stepProc rect)
        let rectsB = stepAccumB [] newRectE
        (pureB (fun rects rect -> 
                    { rectangles = rects
                      currentRec = rect
                    })) <.> rectsB <.> rectB'


  let drawRectangle (Rect ((Vector (x0, y0)), (Vector (x1, y1)))) (gd:GraphicsDevice) = 
        let n_verts = 5
        let random_vert _ = Graphics.VertexPositionColor(Vector3(0.f, 0.f, 0.f), Graphics.Color.White)
        let vertex = Array.init n_verts random_vert
        vertex.[0].Position <- Vector3((float32)x0, (float32)y0, (float32) 0.0)
        vertex.[1].Position <- Vector3((float32)x0, (float32)y1, (float32) 0.0)
        vertex.[2].Position <- Vector3((float32)x1, (float32)y1, (float32) 0.0)
        vertex.[3].Position <- Vector3((float32)x1, (float32)y0, (float32) 0.0)
        vertex.[4].Position <- Vector3((float32)x0, (float32)y0, (float32) 0.0)
        gd.DrawUserPrimitives(PrimitiveType.LineStrip, vertex, 0, n_verts-1)


  let renderer state (gd:GraphicsDevice) = 
        match (state.currentRec, state.rectangles) with
        |(Some r), rects -> drawRectangle r gd 
                            List.iter (fun r -> drawRectangle r gd) rects
        |None, rects -> List.iter (fun r -> drawRectangle r gd) rects

  let renderedGame (game:Game) = 
        let stateB = mainGame game
        (pureB renderer) <.> stateB 


  do use game = new XnaTest2(renderedGame)
     game.Run()


