#light


namespace Xna
 module Main =
  open System
  open FsReactive.Misc
  open FsReactive.FsReactive
  open FsReactive.Integration
  open FsReactive.Lib
 
  open Microsoft.Xna.Framework
  open Microsoft.Xna.Framework.Graphics
  open Microsoft.Xna.Framework.Input;

  let mousePos (game:Game) () = 
        let (xm, ym) = let ms = Mouse.GetState() in ((float)ms.X, (float)ms.Y)
        let (xw, yw, ww, hw) = let cb = game.Window.ClientBounds
                               ((float)cb.X, (float)cb.Y, (float)cb.Width, (float)cb.Height)
        let (xmo, ymo) = (2.0*(xm)/ww-1.0, 1.0-2.0*(ym)/hw)
        if (xmo < -1.0 || 1.0 < xmo || ymo < -1.0 || 1.0 < ymo) 
        then None
        else Some (xmo, ymo)

            
  type  XnaTest2(game : Game -> (GraphicsDevice -> unit) Behavior) as this =
    inherit Game()

    let mutable cameraView : Matrix = Matrix.Identity
    let mutable cameraProjection : Matrix = Matrix.Identity

    let mutable  graphics : GraphicsDeviceManager = null

    let mutable effect : Graphics.Effect = null

    let mutable vertexDeclaration : Graphics.VertexDeclaration = null

    let mutable behavior : unit-> (GraphicsDevice -> unit) Behavior = fun () -> game this
    let mutable time : float = 0.0

    let rand = System.Random()

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
        gd.Clear Graphics.Color.Black
   
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

        member g.getMousePos() = 
            let (xm, ym) = let ms = Mouse.GetState() in ((float)ms.X, (float)ms.Y)
            let (xw, yw, ww, hw) =  let cb = g.Window.ClientBounds
                                    ((float)cb.X, (float)cb.Y, (float)cb.Width, (float)cb.Height)
            let (xmo, ymo) = (2.0*(xm)/ww-1.0, 1.0-2.0*(ym)/hw)
            if (xmo < -1.0 || 1.0 < xmo || ymo < -1.0 || 1.0 < ymo) 
            then None
            else Some  (xmo, ymo)
    end
