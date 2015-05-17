open System
open System.IO
open System.Drawing
open System.Threading
open System.Windows.Forms
open System.Collections.Concurrent
open System.Collections.Generic
open System.Drawing
open Microsoft.FSharp.Control
open ImageProcessing
open System
open FSharp.Actor
open StateBuilder.StateBuilderModule

[<EntryPoint>]
let main argv = 
    
    let state = StateBuilder()

    // Primitive functions for getting and setting state
    let GetState          = StatefulFunc (fun state -> state, state)
    let SetState newState = StatefulFunc (fun prevState -> (), newState)  

    let Add x =
        state { 
            let! currentTotal, history = GetState
            do! SetState (currentTotal + x, (sprintf "Added %d" x) :: history) 
        }
    
    let Subtract x =
        state {
            let! currentTotal, history = GetState
            do! SetState (currentTotal - x, (sprintf "Subtracted %d" x) :: history) 
        }
    
    let Multiply x =
        state { 
            let! currentTotal, history = GetState
            do! SetState (currentTotal * x, (sprintf "Multiplied by %d" x) :: history) 
        }

    let Divide x =
        state {
            let! currentTotal, history = GetState
            do! SetState (currentTotal / x, (sprintf "Divided by %d" x) :: history)
        }

    // ----------------------------------------------------------------------------

    // Define the StatefulFunc we will use, no need to thread
    // the state parameter through each function.
    let calculatorActions =
        state {
            do! Add 2
            do! Multiply 10
            do! Divide 5
            do! Subtract 8
        
            return "Finished" 
        }

    // Now run our SatefulFunc passing in an intial state
    let sfResult, finalState = Run calculatorActions (0, [])

    printfn "Result %s - state %A" sfResult finalState

    Console.ReadLine() |> ignore
    0 
