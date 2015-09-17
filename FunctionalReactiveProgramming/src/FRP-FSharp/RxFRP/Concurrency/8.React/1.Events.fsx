//5-7
// Click events accumulate separately. 
open System
open System.Drawing
open System.Windows.Forms
open System.Threading
open System.IO
open System.Windows.Forms

let fnt = new Font("Calibri", 24.0f)
let lbl = new System.Windows.Forms.Label(Dock = DockStyle.Fill, 
                                         TextAlign = ContentAlignment.MiddleCenter, 
                                         Font = fnt)

let form = new Form(ClientSize = Size(200, 100), Visible = true)
do form.Controls.Add(lbl)

let regsiter ev =  
    ev   
    |> Event.map    (fun _ -> DateTime.Now)                                    // Create events carrying the current time
    |> Event.scan   (fun (_, currentStamp : DateTime) lastStamp ->             // Remembers the last time click was accepted
                        if ((lastStamp - currentStamp).TotalSeconds > 2.0)     // When the time is more than two seconds...
                            then (4, lastStamp)                                // .. we return 4 and the last time click was accepted
                            else (1, currentStamp))                            // .. we return 1 and the new current time
                    (0, DateTime.Now) 
    |> Event.map    fst                                                        // Grab the count (discard the time stamp)
    |> Event.scan   (+) 0                                                      // Sum the yielded numbers 
    |> Event.map    (sprintf "Clicks: %d")                                     // Format the output as a string 
    |> Event.add    lbl.set_Text                                               // Display the result    

regsiter lbl.MouseDown

//  asynchronous loop
let rec loop count = 
    async { 
        // Wait for the next click
        let! ev = Async.AwaitEvent lbl.MouseDown
        lbl.Text <- sprintf "Clicks: %d" count
        do! Async.Sleep 1000        
        return! loop <| count + 1
    }
let start = Async.StartImmediate <| loop 1


form.Show()
