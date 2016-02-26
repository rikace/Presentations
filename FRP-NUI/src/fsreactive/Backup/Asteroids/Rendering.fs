
namespace Asteroids

 module Rendering = 
    
  open Microsoft.Xna.Framework
  open Microsoft.Xna.Framework.Graphics
  open Common.Vector
  open System
  open Asteroids.Data
  
  let drawShip (gd:GraphicsDevice) scale = 
        let pts = List.map (fun (x,y) -> (x/40.0, y/40.0)) [(-2.0, 2.0);(4.0, 0.0);(-2.0, -2.0);(0.0, 0.0);(-2.0, 2.0)]
        let n_verts = List.length pts
        let random_vert _ =  Graphics.VertexPositionColor(Vector3(0.f, 0.f, 0.f), Graphics.Color.White)
        let vertex = Array.init n_verts random_vert

        let jet_n_verts = 3
        let jet_random_vert _ =  Graphics.VertexPositionColor(Vector3(0.f, 0.f, 0.f), Graphics.Color.White)
        let jet_vertex = Array.init jet_n_verts jet_random_vert

        let iter f pts (vertex: VertexPositionColor[]) = 
                List.iteri (fun i (x,y) -> let (x', y') = f x y
                                           vertex.[i].Position <- Vector3((float32)x', (float32)y', (float32) 0.0)) pts
        let draw (gd:GraphicsDevice) x y angle jet = 
            let cosa = Math.Cos angle                 
            let sina = Math.Sin angle 
            let f x' y' = x + (x' * cosa - y' * sina) * scale, y + (x' * sina + y' * cosa) * scale           
            iter f pts vertex
            gd.DrawUserPrimitives(PrimitiveType.LineStrip, vertex, 0, n_verts-1)
            // jet
            match jet with
            |Some s ->  let l = match s with
                                |JetSmall -> -2.0
                                |JetMedium -> -3.0
                                |JetBig -> -4.0
                        let jetpts = List.map (fun (x,y) -> (x/40.0, y/40.0)) [(-1.0, 1.0);(l, 0.0);(-1.0, -1.0)]
                        iter f jetpts jet_vertex
                        gd.DrawUserPrimitives(PrimitiveType.LineStrip, jet_vertex, 0, jet_n_verts-1)
            |None -> ()
        draw gd       

  let drawShip' (gd:GraphicsDevice) (Ship (_, Vector(x, y), angle, jet)) = drawShip gd 1.0 x y angle jet
  
  
  let drawExplodingShip (gd:GraphicsDevice)  = 
        let pts = List.map (fun ((x,y), (x1, y1)) -> ((x/40.0, y/40.0), (x1/40.0, y1/40.0)))
                                                           [((-2.0, 2.0), (4.0, 0.0));
                                                            ((4.0, 0.0), (-2.0, -2.0));
                                                            ((-2.0, -2.0), (0.0, 0.0));
                                                            ((0.0, 0.0), (-2.0, 2.0))]
        let dirs = List.map (fun (x,y) -> (x/40.0, y/40.0)) [(1.0, 1.0); (1.0, -1.0); (-1.0, -1.0); (-1.0, 1.0)]

        let n_verts = List.length pts * 2
        let random_vert _ =  Graphics.VertexPositionColor(Vector3(0.f, 0.f, 0.f), Graphics.Color.White)
        let vertex = Array.init n_verts random_vert

        let iter f pts (vertex: VertexPositionColor[]) = 
                List.iteri (fun i (x,y) -> let (x', y') = f x y
                                           vertex.[i].Position <- Vector3((float32)x', (float32)y', (float32) 0.0)) pts
        let draw (gd:GraphicsDevice) x y angle scale = 
            if scale > 0.0
            then 
                let pts' = List.map2 (fun ((x,y), (x1, y1)) (dirx, diry) -> ((x+dirx*scale, y+diry*scale), (x1+dirx*scale, y1+diry*scale))) pts dirs
                let pts'' = List.fold (fun acc (a,b) -> acc @ [a;b] ) [] pts'
                let cosa = Math.Cos angle                 
                let sina = Math.Sin angle 
                let f x' y' = x + (x' * cosa - y' * sina) , y + (x' * sina + y' * cosa)            
                iter f pts'' vertex
                gd.DrawUserPrimitives(PrimitiveType.LineList, vertex, 0, n_verts/2)
        draw gd       
  
  let drawExplodingShip' (gd:GraphicsDevice) (DestroyedShip ((Vector(x, y), angle, scale))) = drawExplodingShip gd x y angle scale
  
  let drawMeteor (gd:GraphicsDevice)  = 
        let angles = Seq.toList (seq{for i in 1 .. 11 -> Math.PI*2.0*((float)i)/10.0})
        let ptsB = List.map (fun a-> (Vector.rot Vector.unit a) * (2.0/20.0)) angles
        let ptsM = List.map (fun a-> (Vector.rot Vector.unit a) * (2.0/30.0)) angles
        let ptsS = List.map (fun a-> (Vector.rot Vector.unit a) * (2.0/40.0)) angles
        let n_verts = List.length ptsB
        let random_vert _ =  Graphics.VertexPositionColor(Vector3(0.f, 0.f, 0.f), Graphics.Color.White)
        let vertex = Array.init n_verts random_vert
        let iter f size = List.iteri (fun i (Vector(x,y)) -> let (x', y') = f x y
                                                             vertex.[i].Position <- Vector3((float32)x', (float32)y', (float32) 0.0))        
                                     (match size with
                                      |MeteorSize.Big -> ptsB
                                      |MeteorSize.Medium -> ptsM
                                      |MeteorSize.Small -> ptsS)
        let draw (gd:GraphicsDevice) x y size = 
            let f x' y' = x + x', y + y'          
            iter f size
            gd.DrawUserPrimitives(PrimitiveType.LineStrip, vertex, 0, n_verts-1)
        draw  gd   

  let drawMeteor' (gd:GraphicsDevice) (Meteor (_, Vector(x, y), msize)) = drawMeteor gd x y msize

  let drawShield (gd:GraphicsDevice) = 
        let angles = Seq.toList (seq{for i in 1 .. 31 -> Math.PI*2.0*((float)i)/30.0})
        let pts = List.map (fun a-> (Vector.rot Vector.unit a) * 0.12) angles
        let n_verts = List.length pts
        let random_vert _ =  Graphics.VertexPositionColor(Vector3(0.f, 0.f, 0.f), Graphics.Color.White)
        let vertex = Array.init n_verts random_vert
        let iter f  = List.iteri (fun i (Vector(x,y)) -> let (x', y') = f x y
                                                         vertex.[i].Position <- Vector3((float32)x', (float32)y', (float32) 0.0))       
                                     
        let draw (gd:GraphicsDevice) x y  = 
            let f x' y' = x + x', y + y'          
            iter f pts
            gd.DrawUserPrimitives(PrimitiveType.PointList, vertex, 0, n_verts)
        draw  gd   
 
  let drawShield' (gd:GraphicsDevice) shieldOn ship = 
    match (shieldOn, ship) with
    |(true, Some (Ship (_, Vector(x, y), _, _))) -> drawShield gd x y 
    |_ ->()


  let drawBullet (gd:GraphicsDevice) = 
        let n_verts = 1
        let random_vert _ =  Graphics.VertexPositionColor(Vector3(0.f, 0.f, 0.f), Graphics.Color.White)
        let vertex = Array.init n_verts random_vert
        let draw (gd:GraphicsDevice) x y =  
             vertex.[0].Position <- Vector3((float32)x, (float32)y, (float32) 0.0)
             gd.DrawUserPrimitives(PrimitiveType.PointList, vertex, 0, n_verts)
        draw gd                 

  let drawBullet' (gd:GraphicsDevice) (Bullet (_, Vector(x, y))) =  drawBullet gd x y
 
  let drawRemainingShips (gd:GraphicsDevice) n = 
    let xoffset = seq { for i in 0 .. n-1 -> (float) i * 0.05 }
    Seq.iter (fun i -> drawShip gd 0.2 (-0.95 + i ) 0.9 (Math.PI/2.0) None) xoffset
