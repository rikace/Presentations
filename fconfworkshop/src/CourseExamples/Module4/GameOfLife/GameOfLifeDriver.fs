module GameOfLifeDriver

open System
open System.Collections.Generic
open System.Linq
open GameOfLifeLogic
open GameOfLifeUI

let getRandomBool =
    let random = Random(int System.DateTime.Now.Ticks)
    fun () -> random.Next() % 2 = 0

let run() =
    let updateAgent = updateAgent()

    let cells = seq {
        for x = 0 to grid.Width - 1 do
            for y = 0 to grid.Height - 1 do
                yield (x,y), createCell ({x=x;y=y}) (getRandomBool()) updateAgent } |> dict

    let neighbours (location:Location) = seq {
        for x = location.x - 1 to location.x + 1 do
            for y = location.y - 1 to location.y + 1 do
                if x <> location.x || y <> location.y then
                    yield cells.[((x + grid.Width) % grid.Width, (y + grid.Height) % grid.Height)] }

    applyGrid (fun x y ->
            let agent = cells.[(x,y)]
            let neighbours = neighbours {x=x;y=y} |> Seq.toList
            agent (Neighbours(neighbours |> List.map (fun n -> n))))

    let updateView() =
        updateAgent.Post(UpdateView.Reset)
        cells.Values.AsParallel().ForAll(fun cell -> cell (CellMessage.Reset))

    do updateAgent.Start()

    let timer = new System.Timers.Timer(200.)
    let dispose = timer.Elapsed |> Observable.subscribe(fun _ -> updateView())
    timer.Start()
    dispose