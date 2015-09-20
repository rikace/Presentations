#r "System.Windows.Forms"
#r "System.Drawing"

open System
open System.Windows.Forms
open System.Drawing

[<Literal>]
let parts = 25

let arrParts = [|1..parts|]

let shaffle (ls:int array) =    
    let rnd = Random()
    let max = ls.Length - 1
    let swap (buff:int array) (a,b) = 
        let temp = buff.[b]
        buff.[b] <- buff.[a]
        buff.[a] <- temp 
    let getRandomNumber buff max =
        let random = rnd.Next(max)
        swap buff (random, max)
        buff.[max]
    [| for i = max downto 0 do yield getRandomNumber ls i |]

let listofWinners = shaffle arrParts

    
let form = new Form(Visible = true, Text = "DC F# User Group", 
                    TopMost = true, Size = Size(600,600))

let textBox = 
    new RichTextBox(Dock = DockStyle.Fill, Text = "?", 
                    Font = new Font("Lucida Console",158.0f,FontStyle.Bold),
                    ForeColor = Color.DarkBlue)
let index = ref 0
form.Controls.Add(textBox)

let show() =         
    textBox.Text <- sprintf "%40A" (listofWinners.[!index])
    incr(index)
    System.Windows.Forms.Application.DoEvents()



