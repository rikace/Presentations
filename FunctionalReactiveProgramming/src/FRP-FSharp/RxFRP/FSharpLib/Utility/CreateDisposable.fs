namespace Easj360FSharp 

open System

module CreateDisposable =
  
    let createDisposable f =
        {
            new IDisposable with
                member x.Dispose() = f()
        }
 
    let changeColor color () = 
        let current = Console.ForegroundColor
        Console.ForegroundColor <- color
        createDisposable (fun () -> Console.ForegroundColor <- current)

    let red = changeColor ConsoleColor.Red
    let green = changeColor ConsoleColor.Green

//    using (red()) (fun _ ->  printfn "This is red : %A" [1 .. 3])
//    using (green()) (fun _ -> printfn "This is %A : %A" ConsoleColor.Green [1 .. 3])

