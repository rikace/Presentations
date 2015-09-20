// Learn more about F# at http://fsharp.net

open Microsoft.Xna.Framework
open Microsoft.Xna.Framework.Graphics

type XnaGame() as this =
    inherit Game()
    
    do this.Content.RootDirectory <- "XnaGameContent"
    let graphicsDeviceManager = new GraphicsDeviceManager(this)

    let mutable sprite : Texture2D = null
    let mutable spriteBatch : SpriteBatch = null
    let mutable x = 0.f
    let mutable y = 0.f
    let mutable dx = 4.f
    let mutable dy = 4.f

    override game.Initialize() =
        graphicsDeviceManager.GraphicsProfile <- GraphicsProfile.HiDef
        graphicsDeviceManager.PreferredBackBufferWidth <- 640
        graphicsDeviceManager.PreferredBackBufferHeight <- 480
        graphicsDeviceManager.ApplyChanges() 
        spriteBatch <- new SpriteBatch(game.GraphicsDevice)
        base.Initialize()

    override game.LoadContent() =
        sprite <- game.Content.Load<Texture2D>("Sprite")
        
    override game.Update gameTime = 
        if x > 608.f || x < 0.f then dx <- -dx
        if y > 448.f || y < 0.f then dy <- -dy
        
        x <- x + dx
        y <- y + dy


    override game.Draw gameTime = 
        game.GraphicsDevice.Clear(Color.CornflowerBlue)
        spriteBatch.Begin()
        spriteBatch.Draw(sprite, Vector2(x,y), Color.White)
        spriteBatch.End()

let game = new XnaGame()
game.Run()
