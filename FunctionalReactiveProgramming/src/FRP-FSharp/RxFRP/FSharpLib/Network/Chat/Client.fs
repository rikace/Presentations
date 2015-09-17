module Client
#light
open System
open System.ComponentModel
open System.IO
open System.Net.Sockets
open System.Threading
open System.Windows.Forms

let form =
    // create the form
    let form = new Form(Text = "F# Talk Client")

    // text box to show the messages received
    let output = 
        new TextBox(Dock = DockStyle.Fill, 
                    ReadOnly = true, 
                    Multiline = true)
    form.Controls.Add(output)

    // text box to allow the user to send messages
    let input = new TextBox(Dock = DockStyle.Bottom, Multiline = true)
    form.Controls.Add(input)

    // create a new tcp client to handle the network connections
    let tc = new TcpClient() 
    tc.Connect("localhost", 4242) 

    // loop that handles reading from the tcp client
    let load() =
        let run() = 
            let sr = new StreamReader(tc.GetStream())
            while(true) do 
                let text = sr.ReadLine()
                if text <> null && text <> "" then
                    // we need to invoke back to the "gui thread"
                    // to be able to safely interact with the controls
                    form.Invoke(new MethodInvoker(fun () ->
                        output.AppendText(text + Environment.NewLine) 
                        output.SelectionStart <- output.Text.Length))
                    |> ignore

        // create a new thread to run this loop
        let t = new Thread(new ThreadStart(run)) 
        t.Start()

    // start the loop that handles reading from the tcp client
    // when the form has loaded
    form.Load.Add(fun _ -> load())

    let sw = new StreamWriter(tc.GetStream())
    
    // handles the key up event - if the user has entered a line
    // of text then send the message to the server
    let keyUp () = 
        if(input.Lines.Length > 1) then 
            let text = input.Text
            if (text <> null && text <> "") then
                try 
                    sw.WriteLine(text)
                    sw.Flush()
                with err -> 
                    MessageBox.Show(sprintf "Server error\n\n%O" err) 
                    |> ignore
                input.Text <- ""

    // wire up the key up event handler
    input.KeyUp.Add(fun _ -> keyUp ()) 
    
    // when the form closes it's necessary to explicitly exit the app
    // as there are other threads running in the back ground

    form.Closing.Add(fun _ -> 
        Application.Exit()
        Environment.Exit(0))

    // return the form to the top level
    form

// show the form and start the apps event loop
[<STAThread>]
do Application.Run(form)



