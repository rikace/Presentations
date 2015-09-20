#nowarn "40"
#r @"..\..\Reactive.Helpers\bin\Debug\Reactive.Helpers.dll"

open System
open System.Windows.Forms
open System.Drawing
open Reactive.Functional

// ----------------------------------------------------------------------------
// Create WinForms GUI
    
let frm = new Form(ClientSize=Size(600,500))
let fnt = new Font("Arial", 9.75f, FontStyle.Bold)
let lblTime = 
  new Label
   (Location=Point(frm.ClientSize.Width-100,0), ClientSize = Size(100, 22), 
    BackColor=SystemColors.ControlDark, Font = fnt, ForeColor = SystemColors.ControlLight,
    Text = "Time: 20", TextAlign = ContentAlignment.MiddleCenter)
let lblScore = 
  new Label
   (Location=Point(0, 0), ClientSize = Size(100, 22), 
    BackColor=SystemColors.ControlDark, Font = fnt, ForeColor = SystemColors.ControlLight,
    Text = "Score: 0", TextAlign = ContentAlignment.MiddleCenter)
let lblSpacer = 
  new Label
   (Location=Point(100, 0), ClientSize = Size(frm.ClientSize.Width - 200, 22), 
    BackColor=SystemColors.ControlDark)
let picSmiley = 
  new PictureBox
   (SizeMode = PictureBoxSizeMode.AutoSize, Location=Point(75, 75),
    Image = Bitmap.FromFile(__SOURCE_DIRECTORY__ + "\smiley.png"))

frm.Controls.Add(lblTime)
frm.Controls.Add(lblScore)
frm.Controls.Add(lblSpacer)
frm.Controls.Add(picSmiley)

// ----------------------------------------------------------------------------
// Clicking on the smiley & testing whether it was hit & displaying score

let cns a _ = a

let eClicked = 
  picSmiley.MouseDown
  |> Reactive.freeable 
  |> Reactive.filter (fun md -> 
      md.Button = MouseButtons.Left &&
       (pown (float picSmiley.Width / 2. - float md.X) 2 +
        pown (float picSmiley.Height / 2. - float md.Y) 2 < 2250.0 ))
  |> Reactive.sumBy (cns 1)
  |> Reactive.pass (sprintf "Score: %d" >> lblScore.set_Text)
  |> Reactive.map (cns ())
  
// ----------------------------------------------------------------------------
// Moves the smiley every time the user clicked on it or after specified time
let rnd = new Random()
let eMoveSmiley =
  Reactive.After(1000.0, ())
  |> Reactive.switchRecursive (fun () ->
      Reactive.Merge(eClicked, Reactive.After(600.0, ())))
  |> Reactive.map (fun _ -> 
      Point(rnd.Next(frm.ClientSize.Width - picSmiley.Width),        
            20 + rnd.Next(frm.ClientSize.Height - picSmiley.Height - 20)) )
  |> Reactive.pass picSmiley.set_Location            

// ----------------------------------------------------------------------------
// Count-down timer, shows the time on a label
let start = DateTime.Now
Reactive.Repeatedly(1000.0)
  |> Reactive.map (fun e -> 20.0 - e.Subtract(start).TotalSeconds)
  |> Reactive.pass (int >> sprintf "Time: %d" >> lblTime.set_Text)
  |> Reactive.filter ((>) 0.0)
  |> Reactive.first
  |> Reactive.listen(fun _ -> 
        eClicked.Stop()
        eMoveSmiley.Stop()
        MessageBox.Show("Game over!\n" + lblScore.Text) |> ignore )
        

frm.Show()  
