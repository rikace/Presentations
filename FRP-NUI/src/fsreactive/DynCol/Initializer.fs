namespace Initializer
open System
open FsFRPLib.Core
open FsFRPLib
open Common.Random
open Common.Vector

[<AutoOpen>]
module MainGame=
  open System
  open FsFRPLib.Core
  open FsFRPLib
 
  open Microsoft.Xna.Framework
  open Microsoft.Xna.Framework.Graphics
  open Microsoft.Xna.Framework.Input;

  type  GameInitializer(game : Game -> (GraphicsDevice -> unit) Behavior) as this =
    inherit Game()

    let mutable cameraView : Matrix = Matrix.Identity
    let mutable cameraProjection : Matrix = Matrix.Identity
    let mutable  graphics : GraphicsDeviceManager = null
    let mutable effect : Graphics.Effect = null
    let mutable vertexDeclaration : Graphics.VertexDeclaration = null
    let mutable behavior : unit-> (GraphicsDevice -> unit) Behavior = fun () -> game this
    let mutable time : float = 0.0
    do
        graphics <- new GraphicsDeviceManager(this)


    override this.Initialize() =
        base.Initialize()
        this.IsMouseVisible <- true
        let gd = graphics.GraphicsDevice
        effect <- new Graphics.BasicEffect(gd, null)
        let elts = Graphics.VertexPositionColor.VertexElements
        vertexDeclaration <- new Graphics.VertexDeclaration(gd, elts)
    
        let cameraPos = new Vector3((float32)0.0, (float32)0.0, (float32)5.0);

        cameraView <- Matrix.CreateLookAt(cameraPos, Vector3.Zero, Vector3.Up) ;
        cameraProjection <- Matrix.CreatePerspectiveFieldOfView(
                                    MathHelper.PiOver4,
                                    (float32) this.Window.ClientBounds.Width /(float32)this.Window.ClientBounds.Height,
                                    (float32)1.0, (float32)100.0) 
 
    override this.Draw gameTime =
        let (gd:GraphicsDevice) = graphics.GraphicsDevice
        gd.VertexDeclaration <- this.VertexDeclaration 
        gd.Clear Graphics.Color.Gray
        let toBehavior (Behavior b) (t : Time) = b t
        do  let t0 = DateTime.Now;
            let nb = (this.Behavior())
            let (renderf, nb) = toBehavior nb (this.Time)
            this.Behavior <- nb
            this.Time <- this.Time + (1.0/60.0)
            effect.Begin()
            for pass in effect.CurrentTechnique.Passes do
                     pass.Begin()
                     renderf gd
                     pass.End()
            effect.End()
    with
    
        member this.Graphics with get() = graphics
        member this.Effect with get() = effect
        member this.VertexDeclaration with get() = vertexDeclaration
        member this.Behavior with get() = behavior and set(value) = behavior <- value
        member this.Time with get() = time and set(value) = time <- value
    end 

[<AutoOpen>]
module Helpers =

//  
//  let waitEvent dt = 
//        let tmax = 2.0
//        let bf t = let tmax = t+dt
//                   let rec bf' t = if (t < tmax) then (None, fun () -> Event bf')
//                                                 else (Some (), fun () -> noneEvent)
//                   (None, fun () -> Event bf')
//        Event bf
         
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


  let rec dyncolB lb evt = 
    let isSome x =  match x with 
                    |None -> false
                    |Some _ -> true
    let toBehavior (Behavior b) (t : Time) = b t
    let toEvent (Event e) = e
    let bf t = let (r,nb) = List.unzip (List.filter (fst >> isSome) 
                                                    (List.map (fun b -> toBehavior b t) lb ))
               let proc () = let l = List.map (fun x -> x()) nb
                             let (re, ne) = toEvent evt t
                             let l' = match re with
                                      |None -> l
                                      |Some b -> b@l
                             dyncolB l' (ne())
               (catOption r, proc)       
    Behavior bf

  and catOption l =
    let rec proc l acc =
        match l with
        |[] -> acc
        |h::t -> match h with
                 |Some x -> proc t (x::acc)
                 |None -> proc t acc
    List.rev (proc l [])

    // float -> float
  let incrSpeed x = 
    let nx = -x * (1.0 + random() / 2.0)
    if (Math.Abs nx) > 30.0 then 30.0 * ((float) (Math.Sign nx))
    else nx