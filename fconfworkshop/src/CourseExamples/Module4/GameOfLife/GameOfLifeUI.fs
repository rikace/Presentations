module GameOfLifeUI

open System.Windows
open System.Windows.Media
open System.Windows.Media.Imaging
open System.Threading
open System.Collections.Generic
open GameOfLifeLogic

let rect = Int32Rect(0,0,grid.Width, grid.Height)
let source = WriteableBitmap(grid.Width, grid.Height, 96., 96., PixelFormats.Gray8, BitmapPalette([|Colors.Black; Colors.White|]))
let image = Controls.Image(Source=source,Stretch=Stretch.Uniform)

let updateAgent () =
    let ctx = SynchronizationContext.Current
    let pixels = Array.zeroCreate<byte> (size*size)
    let agent = new Agent<UpdateView>(fun inbox ->
        let rec loop agentStates = async {
            let! msg = inbox.Receive()
            match msg with
            | Reset -> return! loop (Dictionary<Location, bool>(HashIdentity.Structural))
            | Update(alive, location) ->
                agentStates.[location] <- alive
                if agentStates.Count = gridProduct then
                    // TASK
                    // complete applyGrid function to change the state of the cell
                    // alive or death according if the cell coordinate exists, which mean the cell is alive
                    applyGrid (fun x y ->
                            match agentStates.TryGetValue({x=x;y=y}) with
                            | true, s when s = true ->
                                pixels.[x+y*size] <- byte 128
                            | _ -> pixels.[x+y*size] <- byte 0)
                    do! Async.SwitchToContext ctx
                    source.WritePixels(rect, pixels, size, 0)
                    do! Async.SwitchToThreadPool()
                return! loop agentStates
        }
        loop (Dictionary<Location, bool>(HashIdentity.Structural)))
    agent