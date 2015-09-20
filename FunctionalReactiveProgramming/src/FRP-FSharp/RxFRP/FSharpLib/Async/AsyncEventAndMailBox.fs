namespace Easj360FSharp 


module AsyncEventAndMailBox =

    module Version1 = 

      // Creating user interface and drawing utility (F#)
      open System
      open System.Drawing
      open System.Windows.Forms
           
      let form = new Form(ClientSize=Size(800, 600), Visible=true, Text = "Test 1")

      // Points are represented as tuples
      let drawRectangle(clr, (x1, y1), (x2, y2)) = 
        use gr = form.CreateGraphics()
        use br = new SolidBrush(clr)
        // Calculate upper left and lower right point
        let left, top = min x1 x2, min y1 y2
        let width, height = abs(x1 - x2), abs(y1 - y2)
        // Clear the window using white color
        gr.FillRectangle(Brushes.White, form.ClientRectangle)
        gr.FillRectangle(br, Rectangle(left, top, width, height))
  
  
      // Workflow for drawing rectangles  
      let rec drawingLoop(clr, from) = async {
        // Wait for the next mouse action
        let! move = Async.AwaitEvent(form.MouseMove)
        if (move.Button &&& MouseButtons.Left) = MouseButtons.Left then
          // Refresh rectangle and continue in the 'Drawing' state
          drawRectangle(clr, from, (move.X, move.Y))
          return! drawingLoop(clr, from)
        else  
          // Return end location to the 'Waiting' state
          return (move.X, move.Y) }

      let waitingLoop() = async {
        let clr = Color.IndianRed
        // Repeat the loop after drawing finishes
        while true do
          let! down = Async.AwaitEvent(form.MouseDown)
          let downPos = (down.X, down.Y)
          if (down.Button &&& MouseButtons.Left) = MouseButtons.Left then
            // Transition to the 'Drawing' state
            let! upPos = drawingLoop(clr, downPos)
            do printfn "Drawn rectanlge (%A, %A)" downPos upPos }

      // Note: You can choose a version to run below...
      let Run() =
        // Start the processing by waiting for MouseDown
        Async.Start(waitingLoop())
        Application.Run(form)
        form.Show()

// Storing state in reactive applications

    module Version2 =   
      open System
      open System.Drawing
      open System.Windows.Forms
           
      let form = new System.Windows.Forms.Form(ClientSize=Size(200, 100), Visible=true)
      //The type representing messages (F#)
  
      // Type alias for representing rectangles
      type RectData = (Color * (int * int) * (int * int))
      type DrawingMessage = 
        // Messages for updating the state
        | AddRectangle of RectData
        | SetColor of Color
        // Messages for reading the state
        | GetColor of AsyncReplyChannel<Color>
        | GetRectangles of AsyncReplyChannel<list<RectData>>

      //Creating the mailbox processor

      // Starts by running the given function
      let state = MailboxProcessor.Start(fun mbox ->
        let rec loop(clr, rects) = async { 
          // Asynchronously wait for the next message
          let! msg = mbox.Receive()
          match msg with
          | SetColor(clr) -> return! loop(clr, rects)
          | AddRectangle(rc) -> 
              form.Invalidate()
              // Update the state during recursive call
              return! loop(clr, rects@[rc]) 
          | GetColor(chnl) -> 
              // Return the current color
              chnl.Reply(clr)
              return! loop(clr, rects)
          | GetRectangles(chnl) -> 
              // Return the current list of rectangles
              chnl.Reply(rects)
              return! loop(clr, rects) }
        // Start with initial color and an empty list
        loop(Color.Black, []) )
    
      // --------------------------------------------------------------------------
      // Communicating using messages
  
      let drawRectangle(clr, (x1, y1), (x2, y2)) = 
        use gr = form.CreateGraphics()
        use br = new SolidBrush(clr)
        let left, top = min x1 x2, min y1 y2
        let width, height = abs(x1 - x2), abs(y1 - y2)
        // Note: We don't erase the form here!
        gr.FillRectangle(br, Rectangle(left, top, width, height))

      // Utility function for drawing rectanlges
      let redrawWindow(rectangles) =
        use gr = form.CreateGraphics()
        gr.FillRectangle(Brushes.White, form.ClientRectangle)
        for r in rectangles do
          drawRectangle r

      // Changes in the drawing process  
      let rec drawingLoop(clr, from) = async {
        let! move = Async.AwaitEvent(form.MouseMove)
        if (move.Button &&& MouseButtons.Left) = MouseButtons.Left then
          // Get the list with existing rectangles
          let! rects = state.PostAndAsyncReply(GetRectangles)
          // Draw all rectangles including the new one
          redrawWindow(rects)
          drawRectangle(clr, from, (move.X, move.Y))
          return! drawingLoop(clr, from)
        else  
          return (move.X, move.Y) }

      let waitingLoop() = async {
        while true do
          let! down = Async.AwaitEvent(form.MouseDown)
          let downPos = (down.X, down.Y)
          if (down.Button &&& MouseButtons.Left) = MouseButtons.Left then        
            // Get the selected color
            let! clr = state.PostAndAsyncReply(GetColor) 
            let! upPos = drawingLoop(clr, downPos)
            // Add the newly created rectangle
            state.Post(AddRectangle(clr, downPos, upPos)) }

      // --------------------------------------------------------------------------
      // Adding the user interfce
  
      let Run() =
        // Create the GUI controls
        let btnColor = new ToolStripButton("Choose color")
        let tools = new ToolStrip()
        tools.Items.Add(btnColor) |> ignore
        form.Controls.Add(tools)
    
        // Listing 16.16 Implementing the user interface 
        btnColor.Click.Add(fun _ ->
          // Show dialog for color selection
          use dlg = new ColorDialog()
          if (dlg.ShowDialog() = DialogResult.OK) then 
          // Send the selected color to mailbox processor
            state.Post(SetColor(dlg.Color)) )
    
        form.Paint.Add(fun e ->
          // Get a list with current rectangles
          let rects = state.PostAndReply(GetRectangles)
          redrawWindow(rects) )    

        // Start the process for drawing rectangles
        Async.Start(waitingLoop())
        Application.Run(form)

(*    module Version3 =   
      open System
      open System.Drawing
      open System.Windows.Forms
           
      let form = new Form(ClientSize=Size(800, 600))
  
      type RectData = (Color * (int * int) * (int * int))

      type DrawingMessage = 
        | AddRectangle of RectData
        | SetColor of Color
        | GetColor of AsyncReplyChannel<Color>
        | GetRectangles of AsyncReplyChannel<list<RectData>>

      // --------------------------------------------------------------------------
      // Listing 16.17 Encapsulating mailbox processor into a type

      type DrawingState() = 
        // Private mailbox processor value
        let state = MailboxProcessor.Start(fun mbox ->
          let rec loop(clr, rects) = async { 
            let! msg = mbox.Receive()
            match msg with
            | SetColor(clr) -> return! loop(clr, rects)
            | AddRectangle(rc) -> 
                form.Invalidate()
                return! loop(clr, rects@[rc]) 
            | GetColor(chnl) -> 
                chnl.Reply(clr)
                return! loop(clr, rects)
            | GetRectangles(chnl) -> 
                chnl.Reply(rects)
                return! loop(clr, rects) }
          loop(Color.Black, []) )
    
        // Non-blocking operations without return value
        member x.SetColor(clr) = 
          state.Post(SetColor(clr))
        member x.AddRectangle(rc) = 
          state.Post(AddRectangle(rc))    
        // Asynchronous operations for reading the state
        member x.AsyncGetRectangles() = 
          state.PostAndAsyncReply(GetRectangles)
        member x.AsyncGetColor() = 
          state.PostAndAsyncReply(GetColor)
        // Blocking call for obtaining rectangle list
        member x.GetRectangles() = 
          state.PostAndReply(GetRectangles)
  
      // Initialize the mailbox processor
      let state = new DrawingState()    
  
      // --------------------------------------------------------------------------
  
      let drawRectangle(clr, (x1, y1), (x2, y2)) = 
        use gr = form.CreateGraphics()
        use br = new SolidBrush(clr)
        let left, top = min x1 x2, min y1 y2
        let width, height = abs(x1 - x2), abs(y1 - y2)
        gr.FillRectangle(br, Rectangle(left, top, width, height))

      let redrawWindow(rectangles) =
        use gr = form.CreateGraphics()
        gr.FillRectangle(Brushes.White, form.ClientRectangle)
        for r in rectangles do
          drawRectangle r

      // --------------------------------------------------------------------------
  
      let rec drawingLoop(clr, from) = async {
        let! ev = Async.AwaitEvent(form.MouseMove, form.KeyDown)
        match ev with
        | Choice1Of2(move) when (move.Button &&& MouseButtons.Left) = MouseButtons.Left ->
            let! rects = state.AsyncGetRectangles()
            redrawWindow(rects)
            drawRectangle(clr, from, (move.X, move.Y))
            return! drawingLoop(clr, from)
        | Choice1Of2(move) -> return Some(move.X, move.Y)
        | Choice2Of2(key) when key.KeyCode = Keys.Escape -> 
            form.Invalidate(); 
            return None
        | _ -> 
            return! drawingLoop(clr, from) }

      // --------------------------------------------------------------------------

      let waitingLoop() = async {
        while true do
          let! down = Async.AwaitEvent(form.MouseDown)
          let downPos = (down.X, down.Y)
          if (down.Button &&& MouseButtons.Left) = MouseButtons.Left then        
            let! clr = state.AsyncGetColor()
            let! upPos = drawingLoop(clr, downPos)
            if (upPos.IsSome) then
              state.AddRectangle(clr, downPos, upPos.Value) }

      let Run() =
        let btnColor = new ToolStripButton("Choose color")
        let tools = new ToolStrip()
        tools.Items.Add(btnColor) |> ignore
        form.Controls.Add(tools)
    
        btnColor.Click.Add(fun _ ->
          use dlg = new ColorDialog()
          if (dlg.ShowDialog() = DialogResult.OK) then 
            state.SetColor(dlg.Color) )
    
        form.Paint.Add(fun e ->
          redrawWindow(state.GetRectangles()) )    
    
        Async.Spawn(waitingLoop())
        Application.Run(form)

    // --------------------------------------------------------------------------
    // Choose a version of the application which you want to run here:

    [<System.STAThread>]
    do 
      // Version1.Run()
      // Version2.Run()
      Version3.Run()

  *)