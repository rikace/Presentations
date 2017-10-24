module GameOfLifeLogic

open System.Collections.Generic

let size = 100
type Grid = { Width:int; Height:int }
let gridProduct = size * size
let grid = { Width=size; Height=size }
    
//[<Struct>]
type Location = {x:int; y:int}

let applyGrid f =
    for x = 0 to grid.Width - 1 do
        for y = 0 to grid.Height - 1 do f x y

type CellRef = CellMessage -> unit
and CellMessage =
    | NeighbourState of cell:Location * isalive:bool
    | State of cellstate:CellRef
    | Neighbours of cells:CellRef list
    | Reset

type State =
    {
        neighbours:CellRef list
        wasAlive:bool
        isAlive:bool 
    }
    with static member createDeafault isAlive = { neighbours=[];isAlive=isAlive; wasAlive=false; }

//[<Struct>]
type UpdateView =
    | Reset
    | Update of bool * Location
 
type Agent<'a> = MailboxProcessor<'a>

let createCell location alive (updateAgent : Agent<_>) =
    let neighbourStates = Dictionary<Location, bool>()
    let agentCell = 
        Agent<CellMessage>.Start(fun inbox ->
            let rec loop state = async {
                let! msg = inbox.Receive()
                let state =
                    match msg with
                    | CellMessage.Reset ->
                        let msg = State(fun msg -> inbox.Post msg)
                        for cell in state.neighbours do
                            cell msg
                        neighbourStates.Clear()
                        { state with wasAlive=state.isAlive }
                    | Neighbours(neighbours) -> { state with neighbours=neighbours }
                    | State(c) -> c (NeighbourState(location, state.wasAlive))
                                  state
                    // TASK 
                    // Complete the NeighbourState case applying the Game of Life rules to 
                    // change the state of the cell to alive or depth
                    // Suggestion, if the cell does not have 8 neighbors means that the cell is in the corner
                    | NeighbourState(cell, alive) ->
                        neighbourStates.[cell] <- alive
                        if neighbourStates.Count = 8 then
                            let aliveState =
                                match neighbourStates |> Seq.filter(fun (KeyValue(_,v)) -> v) |> Seq.length with
                                | a when a > 3  || a < 2 -> false
                                | 3 -> true
                                | _ -> state.isAlive
                            updateAgent.Post (Update(aliveState, location))
                            { state with isAlive = aliveState }
                        else state
                return! loop state
            }
            loop (State.createDeafault alive ))

    fun msg -> agentCell.Post msg