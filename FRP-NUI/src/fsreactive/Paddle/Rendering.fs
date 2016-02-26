namespace PongGame

 module Rendering = 
 
    
  open Microsoft.Xna.Framework
  open Microsoft.Xna.Framework.Graphics
  open Common.Vector
  open System

  
  let drawBall x y ballRadius (gd:GraphicsDevice)  = 
        let angles = Seq.toList (seq{for i in 1 .. 11 -> Math.PI*2.0*((float)i)/10.0})
        let pts = List.map (fun a-> (Vector.rot Vector.unit a) * ballRadius) angles
        let count = List.length pts        
        let vertex = Array.init count (fun _ ->  VertexPositionColor(Vector3(0.f, 0.f, 0.f), Color.Red))
        let iter f = List.iteri (fun i (Vector(x,y)) -> let (x', y') = f x y
                                                        vertex.[i].Position <- Vector3((float32)x', (float32)y', (float32) 0.0)
                                                        vertex.[i].Color <- Color.Red) pts
        let f x' y' = x + x', y + y'          
        iter f 
        gd.DrawUserPrimitives(PrimitiveType.LineStrip, vertex, 0, count-1)



  let drawPaddle (x:float) (paddleY:float) (paddleHalfLength:float) (gd:GraphicsDevice)  = 
        let vertex = Array.init 2 (fun _ -> VertexPositionColor(Vector3(0.f, 0.f, 0.f), Color.Red))
        vertex.[0].Position <- Vector3((float32)(x-paddleHalfLength), (float32)(paddleY), (float32) 0.0)
        vertex.[1].Position <- Vector3((float32)(x+paddleHalfLength), (float32)(paddleY), (float32) 0.0)
        gd.DrawUserPrimitives(PrimitiveType.LineStrip, vertex, 0, 1)


  let drawPaddle2 (x:float) (paddleY:float) (paddleHalfLength:float) (gd:GraphicsDevice)  = 
       //let random_vert _ =  Graphics.VertexPositionColor(Vector3(0.f, 0.f, 0.f), Graphics.Color.Red)
        let random_vert _ =  VertexPositionColor(Vector3(0.f, 0.f, 0.f), Color.Red)
        let vertex = Array.init 4 random_vert
//        vertex.[0].Position <- Vector3((float32)(x-paddleHalfLength), (float32)(paddleY), (float32) 0.0)
//        vertex.[1].Position <- Vector3((float32)(x+paddleHalfLength), (float32)(paddleY), (float32) 0.0)
        


        vertex.[0] <- VertexPositionColor(Vector3((float32)(x-paddleHalfLength), (float32)(paddleY), (float32) 0.0), Color.Red)
        vertex.[1] <- VertexPositionColor(Vector3((float32)(x+paddleHalfLength), (float32)(paddleY), (float32) 0.0), Color.Red)
        vertex.[2] <- VertexPositionColor(Vector3((float32)(x+paddleHalfLength), (float32)(paddleY), (float32) 0.0), Color.Red)
        vertex.[3] <- VertexPositionColor(Vector3((float32)(x+paddleHalfLength), (float32)(paddleY), (float32) 0.0), Color.Red)
//        //
        vertex.[0].Color <- Color.Red
        vertex.[1].Color <- Color.Red
        
        gd.DrawUserPrimitives(PrimitiveType.TriangleStrip, vertex, 0, 2)
        