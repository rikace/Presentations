open System
open Microsoft.AspNet.SignalR.Client
open System.IO
open System.Windows.Forms
open System.Drawing
open System.Threading
open System.Reactive.Concurrency
open System.Reactive.Threading
open System.Reactive.Linq
open System.Threading.Tasks
open AsyncHelper

type public SignalRClientForm() as form = 
    inherit Form()
    let panel = new Panel()
    let textBox = new TextBox()
    let textArea = new TextBox()
    let btn = new Button()

    let initControls = 
        form.Width <- 300
        form.Height <- 350
        form.Visible <- true
        form.Text <- "SignalR Client"
        form.Margin <- new System.Windows.Forms.Padding(2, 2, 2, 2)
        form.AutoScaleMode <- System.Windows.Forms.AutoScaleMode.Font
        form.AutoScroll <- true

        textBox.Width <- 150
        textBox.Location <- new System.Drawing.Point(5, 5)

        btn.Width <- 50
        btn.Text <- "Send"
        btn.Location <- new System.Drawing.Point(155, 5)

        textArea.Width <- 200
        textArea.Multiline <- true
        textArea.Lines <- [|"";"";"";""|]
        textArea.Height <- 150
        textArea.Location <- new System.Drawing.Point(5, 40)


        panel.AutoScroll <- true
        panel.Anchor <- AnchorStyles.Top ||| AnchorStyles.Left
        panel.Location <- new System.Drawing.Point(0, 0)
        panel.Margin <- new System.Windows.Forms.Padding(2, 2, 2, 2)
        panel.ClientSize <- new Size(500,500)
        panel.ResumeLayout(false)
        panel.PerformLayout()
    
    do 
        form.SuspendLayout()
        initControls
        panel.Controls.Add(textBox)
        panel.Controls.Add(btn)
        panel.Controls.Add(textArea)

        form.Controls.Add(panel)
        form.Load.AddHandler(new System.EventHandler(fun sender e -> form.eventForm_Loading (sender, e)))
        form.ResumeLayout(false)
        form.PerformLayout()
        form.OnLoad(EventArgs.Empty)
    
    member form.eventForm_Loading (sender : obj, e : EventArgs) = 

        let connection = new Connection("http://localhost:9099/signalrConn")

        connection.AsObservable().ObserveOn(WindowsFormsSynchronizationContext.Current)
        |> Observable.add(fun s ->  printfn "message Conn : %s" s
                                    textArea.AppendText(sprintf "%s\n" s))

        btn.Click
            |> Observable.add(fun _ -> 
                    let message = textBox.Text
                    textBox.Text <- ""
                    connection.Send(message) 
                    |> awaitPlainTask 
                    |> Async.StartImmediate)

        connection.Start().Wait()


    
module Main = 
    [<STAThread>]
    do Application.EnableVisualStyles()
       Application.SetCompatibleTextRenderingDefault(false)
       Application.Run(new SignalRClientForm() :> Form)