namespace PongGame

module PongGameInitializer =
  open System
  open FsFRPx.FsFRPx
  open Leap  
  open Microsoft.Xna.Framework
  open Microsoft.Xna.Framework.Graphics
  open Microsoft.Xna.Framework.Input;


  type LeapListenerMessage =
    | Put of Controller
    | Get of AsyncReplyChannel<(float * float) option>

  type LeapListener() =
    inherit Leap.Listener()

    let leapAgent = MailboxProcessor<LeapListenerMessage>.Start(fun inbox ->
            let rec loop point = async {
                let! msg = inbox.Receive()
                match msg with
                | Put(ctrl) -> 
                    // right hand only one finger move
                    use frame = ctrl.Frame()
                    let hand = frame.Hands.Rightmost                
                    // 2D drawing coordinate systems put the origin at the top, left corner of the window
                    // naturally don’t use a z-axis. 
                    // this code maps Leap Motion coordinates to such a 2D system 
                    let finger = hand.Pointables.[0].StabilizedTipPosition
                    let iBox = frame.InteractionBox
                    let normalizedPoint = iBox.NormalizePoint(finger, true)
                    let x = float normalizedPoint.x //* float windowWidth
                    let y = (1. - float normalizedPoint.y) //* float windowHeight
                    return! loop (Some(x,y))
                | Get(reply) -> reply.Reply(point)
                                return! loop point }
            loop None)
    
    member __.GetPosition() = leapAgent.PostAndReply(fun ch -> Get ch)

    override __.OnFrame(ctrl:Controller) =
        leapAgent.Post(Put ctrl)



  type LeapListenerSingleton private() =
     static let controller = new Controller()
     static let listener = new LeapListener()
     static let leapState() = 
        controller.AddListener(listener) |> ignore
        listener

     static let mutable instance = lazy(leapState())
     static member Instance with get() = instance

  // unit -> (float * float) option
  let leapPos (game:Game) () = 
        let getPoint = LeapListenerSingleton.Instance.Value.GetPosition
        let clientBounds =  game.Window.ClientBounds
        let (xm, ym) = 
            let ms = getPoint() 
            match ms with
            | None -> (0., 0.)
            | Some(x,y) ->  ((x * float clientBounds.Width), (y * float clientBounds.Height))
        let (xw, yw, ww, hw) = let cb = clientBounds
                               ((float)cb.X, (float)cb.Y, (float)cb.Width, (float)cb.Height)
        let (xmo, ymo) = (2.0*(xm)/ww-1.0, 1.0-2.0*(ym)/hw)
        if (xmo < -1.0 || 1.0 < xmo || ymo < -1.0 || 1.0 < ymo) 
        then None
        else Some (xmo, ymo)

























            
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
        let cameraPos = new Vector3((float32)0.0, (float32)0.0, (float32)5.0)
        cameraView <- Matrix.CreateLookAt(cameraPos, Vector3.Zero, Vector3.Up)
        cameraProjection <- Matrix.CreatePerspectiveFieldOfView(
                                    MathHelper.PiOver4,
                                    (float32) this.Window.ClientBounds.Width /(float32)this.Window.ClientBounds.Height,
                                    (float32)1.0, (float32)100.0) 
 
    override this.Draw gameTime =
        let (gd:GraphicsDevice) = graphics.GraphicsDevice
        gd.VertexDeclaration <- this.VertexDeclaration 
        gd.Clear Graphics.Color.Gray
        
        do  let time = DateTime.Now;
            let newBeha = (this.Behavior())
            let (renderf, nb) = atB newBeha (this.Time)
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
