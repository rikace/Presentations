namespace Utility 

//#r "PresentationCore.dll"
//#r "PresentationFramework.dll"
//#r "WindowsBase.dll"
//#r "System.Xaml.dll"

open System.Windows
open System.Windows.Media
open System.Windows.Documents

module DiffTool =

    ()
//    type change =
//            | Copy
//            | Insert
//            | Delete
//
//    let ui x = (x :> UIElement)
//
//    let create ctrlss =
//        let grid = Controls.Grid(Margin=Thickness 5.0)
//        for _ in ctrlss do
//        Controls.RowDefinition() |> grid.RowDefinitions.Add
//        for _ in 1..Seq.reduce max (Seq.map Seq.length ctrlss) do
//        Controls.ColumnDefinition() |> grid.ColumnDefinitions.Add
//        Seq.iteri (fun y ctrls ->
//        Seq.iteri (fun x ctrl ->
//            Controls.Grid.SetColumn(ctrl, x)
//            Controls.Grid.SetRow(ctrl, y)
//            grid.Children.Add ctrl |> ignore) ctrls) ctrlss
//        grid
//
//    let memoize f =
//            let m = System.Collections.Generic.Dictionary(HashIdentity.Structural)
//            fun x ->
//              let f_x = ref Unchecked.defaultof<_>
//              if m.TryGetValue(x, f_x) then
//                !f_x
//              else
//                m.[x] <- f x
//                m.[x]
//
//    let lcs (xs: _ []) (ys: _ []) =
//            let rec lcs = memoize (function
//              | 0, _ | _, 0 -> 0
//              | i, j when xs.[i-1] = ys.[j-1] -> 1 + lcs(i-1, j-1)
//              | i, j -> max (lcs(i-1, j)) (lcs(i, j-1)))
//            let rec walk zs = function
//              | 0, _ | _, 0 -> zs
//              | i, j when lcs(i-1, j) = lcs(i, j) -> walk zs (i-1, j)
//              | i, j -> walk (if lcs(i, j-1) < lcs(i, j) then xs.[i-1]::zs else zs) (i, j-1)
//            walk [] (xs.Length, ys.Length)
//        
//    let color = function
//            | Copy -> Media.Brushes.White
//            | Insert -> Media.Brushes.Green
//            | Delete -> Media.Brushes.Red;;
//
//    let auto x =
//            let s = Controls.ScrollViewer(Content=x)
//            s.HorizontalScrollBarVisibility <- Controls.ScrollBarVisibility.Auto
//            s
//
//    let diff (xs: string) (ys: string) =
//            let rec loop zs = function
//              | x::xs, y::ys, d::ds ->
//                  match x=d, y=d with
//                  | true, true -> loop ((Copy, d)::zs) (xs, ys, ds)
//                  | true, false -> loop ((Insert, y)::zs) (x::xs, ys, d::ds)
//                  | false, true -> loop ((Delete, x)::zs) (xs, y::ys, d::ds)
//                  | false, false -> loop ((Insert, y)::(Delete, x)::zs) (xs, ys, d::ds)
//              | x::xs, ys, [] -> loop ((Delete, x)::zs) (xs, ys, [])
//              | [], y::ys, [] -> loop ((Insert, y)::zs) ([], ys, [])
//              | _ -> List.rev zs
//            let xs, ys = xs.Split[|'\n'|], ys.Split[|'\n'|]
//            loop [] (List.ofArray xs, List.ofArray ys, lcs xs ys)
//
//    let update (text1Box: Controls.TextBox) (text2Box: Controls.TextBox) (diffBox: Controls.RichTextBox) _ =
//            let flow = FlowDocument()
//            let para = Paragraph()
//            for c, xs in diff text1Box.Text text2Box.Text do
//              Run(xs+"\n", Background=color c) |> para.Inlines.Add
//            flow.Blocks.Add para
//            diffBox.Document <- flow
//            
//
//    [<System.STAThread>]
//          do
//            let str1 = "Stringa 1"
//            let str2 = "String2"
//            let text1Box = Controls.TextBox(Text=str1, AcceptsReturn=true)
//            let text2Box = Controls.TextBox(Text=str2, AcceptsReturn=true)
//            let diffBox = Controls.RichTextBox()
//            text1Box.TextChanged.Add(update text1Box text2Box diffBox)
//            text2Box.TextChanged.Add(update text1Box text2Box diffBox)
//            update text1Box text2Box diffBox ()
//            let t s = Controls.Label(Content=s) |> ui
//            let pane = Grid.create [[auto text1Box |> ui; auto text2Box |> ui; ui diffBox]]
//            Application().Run(Window(Content=pane))
//            |> ignore

