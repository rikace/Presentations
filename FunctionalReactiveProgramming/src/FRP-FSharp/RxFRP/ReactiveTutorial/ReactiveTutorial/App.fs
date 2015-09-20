module MainApp

open System
open System.Windows
open System.Windows.Controls

open FSharpx

/// Top-level container for all XAML elements.
type MainWindow = XAML<"MainWindow.xaml">

/// State of the Drag Operation
type DragState = { 
    /// Are we currently dragging?
    dragging : bool;

    /// Current position of the drag
    position : Point; 

    /// Drag offset, for taking into account initial drag position
    offset : Point }

/// Is a drag in progress?
let currently_dragging (state : DragState) : bool = state.dragging

/// Retrieves the current position of the drag operation, taking into account
/// the position and the offset.
let get_drag_position (state : DragState) : Point = 
    let diff = state.position - state.offset
    new Point(diff.X, diff.Y)

/// Initial state of the Drag Operation.
let initial_state = { dragging=false; position=new Point(); offset=new Point() }

/// Produces a new DragState from an existing one but with a different dragging flag
let update_dragging (dragging : bool) (op : DragState) : DragState =
    { op with dragging=dragging }

/// Updates the Position of a DragState. If a Drag isn't currently in progress, does nothing.
let update_drag_pos (position : Point) (op : DragState) : DragState =
    if currently_dragging op
    then { op with position=position }
    else op

/// A type of change that can happen to the DragState
type DragChange = 
    /// Start dragging from a certain offset
    | StartDrag of Point 
    /// Stop dragging
    | StopDrag 
    /// Update the drag with a new position
    | UpdatePosition of Point

/// Updates a DragState by applying a DragChange to it, producing a new DragState.
let update_drag_state (state : DragState) (change : DragChange) : DragState =
    match change with
    | StartDrag(offset) -> { state with dragging=true; offset=offset }
    | StopDrag -> update_dragging false state
    | UpdatePosition(pos) -> update_drag_pos pos state

/// Was a mouse event triggered by the left mouse button?
let is_left_click (args : Input.MouseButtonEventArgs) : bool =
    args.ChangedButton = Input.MouseButton.Left

/// From a mouse event, fetch the position of the cursor relative to a IInputElement
let get_mouse_position (relative_to : IInputElement) (args : Input.MouseEventArgs) : Point =
    args.GetPosition relative_to

/// Initialized the application
let loadWindow() =
    let window = MainWindow()

    /// Sets the position of the rectangle
    let set_rect_position (position : Point) : unit =
        Canvas.SetLeft(window.Rectangle, position.X)
        Canvas.SetTop(window.Rectangle, position.Y)

    /// Get the position of the cursor relative to the Canvas
    let get_canvas_position = get_mouse_position window.Canvas

    /// Get the position of the cursor relative to the Rectangle
    let get_rectangle_position = get_mouse_position window.Rectangle

    /// Stream of StartDrag events
    let start_stream = 
        window.Rectangle.MouseDown
        |> Observable.filter is_left_click
        |> Observable.map (get_rectangle_position >> StartDrag)

    /// Stream of StopDrag events
    let stop_stream =
        window.Canvas.MouseUp
        |> Observable.filter is_left_click
        |> Observable.map (fun _ -> StopDrag)

    /// Stream of UpdatePosition events
    let move_stream =
        window.Canvas.MouseMove
        |> Observable.map (get_canvas_position >> UpdatePosition)

    /// Subscription for the entire Drag command
    let subscription =
        Observable.merge start_stream stop_stream |> Observable.merge move_stream

        |> Observable.scan update_drag_state initial_state
        |> Observable.filter currently_dragging
        |> Observable.map get_drag_position
        |> Observable.subscribe set_rect_position
    
    // Return the Root element
    window.Root

(* Entry Point *)
[<STAThread>]
(new Application()).Run(loadWindow()) |> ignore
