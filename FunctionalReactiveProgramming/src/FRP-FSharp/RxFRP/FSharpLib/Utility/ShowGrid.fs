module ShowGrid


//--------------------------------------------------------------------------------
// Instructions
// (1) select all in this file to run
// (2) Right-click and click on "Send to Interactive"
//
// Running this file shows the retrieved (DateTime * float) tuples in a Grid View
//--------------------------------------------------------------------------------
open System.IO
open System.Net
open System.Windows.Forms

//--------------------------------------------------------------------------------
// declare "grid" function to plot the returned (DateTime * float) tuples
//--------------------------------------------------------------------------------
let grid (prices:seq<System.DateTime * float>) =
    let form = new Form(Visible = true, TopMost = true)
    let grid = new DataGridView(Dock = DockStyle.Fill, Visible = true)

    form.Controls.Add(grid)
    grid.DataSource <- prices |> Seq.toArray



//--------------------------------------------------------------------------------
// function to load prices
//--------------------------------------------------------------------------------
let internal loadPrices ticker =
    let url = "http://ichart.finance.yahoo.com/table.csv?s=" + ticker + "&a=00&b=1&c=2011&d=04&e=17&f=2011&g=d&ignore=.csv"
    let req = WebRequest.Create(url)
    use response = req.GetResponse() // "use" keyword calls Dispose() method when it goes out of scope
    use stream = response.GetResponseStream()
    use reader = new StreamReader(stream)
    let csv = reader.ReadToEnd()

    let prices =
        csv.Split([|'\n'|])
        |> Seq.skip 1
        |> Seq.map (fun currLine -> currLine.Split([|','|]))
        |> Seq.filter (fun values -> (values |> Seq.length = 7))
        |> Seq.map (fun values ->
            System.DateTime.Parse(values.[0]),
            (float) values.[6]
        )
    prices


//--------------------------------------------------------------------------------
// Load prices for MSFT ticker and display in the grid.
grid (loadPrices "MSFT")

