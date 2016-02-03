#r "System.Xaml.dll"         
#r "UIAutomationTypes.dll"   
#r "WindowsBase.dll"         
#r "PresentationFramework.dll"
#r "PresentationCore.dll"

open System.Windows
open System.Threading
open System.Windows.Threading

let ui =
  let mk() =
    let wh = new ManualResetEvent(false)
    let application = ref null
    let start() =
      let app = Application()
      application := app
      ignore(wh.Set())
      app.Run() |> ignore
    let thread = Thread start
    thread.IsBackground <- true
    thread.SetApartmentState ApartmentState.STA
    thread.Start()
    ignore(wh.WaitOne())
    !application, thread
  lazy(mk())

let spawn : ('a -> 'b) -> 'a -> 'b =
  fun f x ->                        
    let app, thread = ui.Force()    
    let f _ =                       
      try                           
        let f_x = f x               
        fun () -> f_x               
      with e ->                     
        fun () -> raise e           
    let t = app.Dispatcher.Invoke(DispatcherPriority.Send, System.Func<_, _>(f), null)
    (t :?> unit -> 'b)()                                                              

let make_window() =
  let window = Window()
  window.Title <- "Windows Presentation Foundation example"
  window.Width <- 707.                                     
  window.Height <- 500.                                    
  window.Show()                                            
  window                                                   
                                                           
spawn make_window ()                                       
                                                           
let simple_controls() =                                    
  let panel = Controls.StackPanel()                        
                                                           
  let add control =                                        
    control |> panel.Children.Add |> ignore                
                                                           
  Controls.Button(Content="Button") |> add                 
                                                           
  Controls.Separator() |> add                              
                                                           
  Controls.CheckBox(Content="Check box") |> add            
                                                           
  Controls.Separator() |> add                              


  Controls.RadioButton(Content="Radio button 1", GroupName="A") |> add
  Controls.RadioButton(Content="Radio button 2", GroupName="A") |> add
  Controls.RadioButton(Content="Radio button 3", GroupName="A") |> add

  Controls.Separator() |> add                                         

  Controls.RadioButton(Content="Radio button 1", GroupName="B") |> add
  Controls.RadioButton(Content="Radio button 2", GroupName="B") |> add
  Controls.RadioButton(Content="Radio button 3", GroupName="B") |> add

  Controls.Separator() |> add

  Controls.Label(Content="This is a label") |> add

  Controls.Separator() |> add

  Controls.Frame(Source=System.Uri "http://www.ffconsultancy.com") |> add

  Controls.Separator() |> add

  Controls.GroupBox(Header="Group box", Content="Content") |> add

  Controls.Separator() |> add

  Controls.Expander(Header="Expander", Content="Content") |> add

  Controls.Separator() |> add

  Controls.Label(Content="Hover over here to see a tooltip",
                 ToolTip=Controls.ToolTip(Content="This is the tooltip"))
  |> add

  make_window().Content <- panel

spawn simple_controls ()

let item_controls() =
  let panel = Controls.StackPanel()

  let add control =                  
    control |> panel.Children.Add |> ignore
                                           
  Controls.ListBox(ItemsSource=[1..3]) |> add
  Controls.Separator() |> add

  Controls.ListView(ItemsSource=[1..3]) |> add
  Controls.Separator() |> add                 
  Controls.TabControl(
    ItemsSource=
      [ for i in 1 .. 3 ->
          Controls.TabItem(Header=sprintf "Tab %d" i, Content=i) ])
  |> add
  Controls.Separator() |> add                                      
                                                                   
  Controls.ComboBox(ItemsSource=[1..3]) |> add                     
                                                                   
  make_window().Content <- panel                                   
                                                                   
spawn item_controls ()                                             
                                                                   
let item header items =                                            
  let items =                                                      
    [ for item in items ->                                         
        Controls.MenuItem(Header=item) ]                           
  Controls.MenuItem(Header=header, ItemsSource=items)              
                                                                   
let menu_window() =                                                
  let menu = Controls.Menu()                                       
  menu.ItemsSource <-                                              
    [ item "File" ["New"; "Open"; "Exit"];                         
      item "Edit" ["Cut"; "Copy"; "Paste"];                        
      item "Help" ["Contents"; "About this demo"] ]                
  let panel = Controls.StackPanel()                                
  panel.Children.Add menu |> ignore

  make_window().Content <- panel                                   
                                                                   
spawn menu_window ()                                               
                                                                   
let context_menu_window() =                                        
  let menu = Controls.ContextMenu()                                
  menu.ItemsSource <-                                              
    [ item "File" ["New"; "Open"; "Exit"];                         
      item "Edit" ["Cut"; "Copy"; "Paste"];                        
      item "Help" ["Contents"; "About this demo"] ]                
  make_window().ContextMenu <- menu                                
                                                                   
spawn context_menu_window ()                                       
                                                                   
let rec file_tree dir leaf node =                                  
  seq { for dir in System.IO.Directory.GetDirectories dir do       
          yield file_tree dir leaf node                            
        for file in System.IO.Directory.GetFiles dir do            
          yield leaf file }                                        
  |> node dir                                                      
                                                                   
let treeview_window() =                                            
  let treeview = Controls.TreeView()                               
  treeview.ItemsSource <-                                          
    [ file_tree @"C:\Users\Public\Documents\Installers"            
        (fun x -> Controls.TreeViewItem(Header=x))                 
        (fun x xs -> Controls.TreeViewItem(Header=x, ItemsSource=xs)) ]
  make_window().Content <- treeview                                    
                                                                       
spawn treeview_window ()                                               
                                                                       
type Scene() =               
  inherit UIElement()        
                             
  override this.OnRender dc =
    base.OnRender dc         
    let pen = Media.Pen(Media.Brushes.Red, 0.3)
    for i=0 to 100 do                          
      let x = 4. * float i                     
      let xys = [x, 0.; 400., x; 400. - x, 400.; 0., 400. - x; x, 0.]
      for ((x0, y0), (x1, y1)) in Seq.pairwise xys do                
        dc.DrawLine(pen, Point(x0, y0), Point(x1, y1))               
                                                                     
spawn (fun () -> make_window().Content <- Scene()) ()                
                                                                     
spawn (fun () ->                                                     
  let line_to(x, y) = (Media.LineSegment(Point(x, y), true) :> Media.PathSegment)
  let geometry =                                                                 
    Media.PathGeometry                                                           
      [ for i in 0..100 ->                                                       
          let x = 4. * float i                                                   
          let xys = [400., x; 400. - x, 400.; 0., 400. - x]                      
          Media.PathFigure(Point(x, 0.), Seq.map line_to xys, true) ]            
  let path = Shapes.Path(Data=geometry, Stroke=Media.Brushes.Red, StrokeThickness=0.3)
  let root = Controls.Viewbox(Child=path, Stretch=Media.Stretch.Uniform)              
  make_window().Content <- root) ()                                                   
                                                                                      
                                                                                      
                                                                                      
open System.Windows.Media.Media3D                                                     
                                                                                      
module Icosahedron =                                                                  
  let vertex =                                                                        
    let g = (1.0 + sqrt 5.0) / 4.0                                                    
    let a, b, c = [0.0], [0.5; -0.5], [g; -g]                                         
    seq { for xs, ys, zs in [ a, b, c; b, c, a; c, a, b ] do                          
            for x in xs do                                                            
              for y in ys do                                                          
                for z in zs do                                                        
                  yield (x, y, z) }                                                   
                                                                                      
  let normals, positions =                                                            
    let within d =                                                                    
      d > 0.75 && d < 1.25                                                            
    let vertex = [|for x, y, z in vertex -> Vector3D(x, y, z)|]                       
    let n = vertex.Length                                                             
    let xs =                                                                          
      seq { for i in 0 .. n-3 do                                                      
              let p0 = vertex.[i]                                                     
              for j in i+1 .. n-2 do                                                  
                let p1 = vertex.[j]                                                   
                if within (p1 - p0).Length then                                       
                  for k in j+1 .. n-1 do                                              
                    let p2 = vertex.[k]                                               
                    if within (p2 - p0).Length && within (p2 - p1).Length then        
                      let normal = Vector3D.CrossProduct(p2 - p0, p1 - p0)            
                      normal.Normalize()                                              
                      let d = Vector3D.DotProduct(p0, normal)                         
                      let normal, p0, p1, p2 =                                        
                        if d < 0.0 then normal, p0, p1, p2 else -normal, p0, p2, p1   
                      for p in [p0; p1; p2] do                                        
                        yield normal, Point3D(p.X, p.Y, p.Z) }                        
    Seq.map fst xs, Seq.map snd xs                                                    
                                                                                      
spawn (fun () ->                                                                      
  let mesh = MeshGeometry3D()                                                         
  mesh.Normals <- Vector3DCollection Icosahedron.normals                              
  mesh.Positions <- Point3DCollection Icosahedron.positions                           
  mesh.TriangleIndices <- Media.Int32Collection[0 .. Seq.length Icosahedron.normals - 1]
  let model = GeometryModel3D(mesh, DiffuseMaterial Media.Brushes.White)                
  let root = Media.Media3D.Model3DGroup()                                               
  for color, direction in                                                               
    [ Media.Colors.Red, Vector3D(-1., 0., -1.);                                         
      Media.Colors.Green, Vector3D(-1., -1., -1.);                                      
      Media.Colors.Blue, Vector3D(0., -1., -1.) ] do                                    
    Media.Media3D.DirectionalLight(Color=color, Direction=direction) |> root.Children.Add |> ignore
  model |> root.Children.Add |> ignore                                                             
  let visual = Media.Media3D.ModelVisual3D(Content=root)                                           
  let viewport3d = Controls.Viewport3D()                                                           
  viewport3d.Camera <-                                                                             
    PerspectiveCamera(Position=Point3D(2.5, 2.5, -4.), LookDirection=Vector3D(-2.5, -2.5, 4.), FieldOfView=30.)
  visual |> viewport3d.Children.Add |> ignore                                                                  
  make_window().Content <- viewport3d                                                                          
                                                                                                               
  let time = System.Diagnostics.Stopwatch.StartNew()                                                           
  Media.CompositionTarget.Rendering.Add(fun _ ->                                                               
    let t = float time.Elapsed.TotalSeconds * 30.                                                              
    model.Transform <- RotateTransform3D(AxisAngleRotation3D(Vector3D(0., 1., 0.), t)))) ()


