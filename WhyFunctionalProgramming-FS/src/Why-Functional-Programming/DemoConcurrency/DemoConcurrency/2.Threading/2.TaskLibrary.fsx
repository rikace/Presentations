#load "..\Utilities\ImageHelpers.fsx"
open System
open System.IO
open System.Threading
open System.Threading.Tasks
open System.Drawing
open System.Drawing.Imaging
open ImageHelpers

let task(action) = Task.Factory.StartNew(action)

let startLongRunningTask() =
    Task.Factory.StartNew(fun _ -> 
        let mutable i = 1
        let mutable loop = true
        
        while i <= 10 && loop do
            printfn "%d..." i
            i <- i + 1
            Thread.Sleep(1000)                
        printfn "Complete!"
    )

let t = startLongRunningTask()

// Spawns a new PFX task to resize an image
let spawnTask filePath = 
    let taskBody = new Action(fun () -> resizeImage (640, 480) filePath)
    Task.Factory.StartNew(taskBody)
        
let imageFiles = Directory.GetFiles(__SOURCE_DIRECTORY__ + "\\..\\Data\\Images", "*.jpg")

// Spawning resize tasks
let resizeTasks = imageFiles |> Array.map spawnTask
Task.WaitAll(resizeTasks)


let spawnToGrayTask filePath = 
    let taskBody = new Action(fun () -> toGray filePath)
    Task.Factory.StartNew(taskBody)

let toGrayTasks = imageFiles |> Array.map spawnToGrayTask
Task.WaitAll(toGrayTasks)


let spawnToGrayParalleTask filePath = 
    let taskBody = new Action(fun () -> toGrayParallel filePath)
    Task.Factory.StartNew(taskBody)

let toGrayParallelTasks = imageFiles |> Array.map spawnToGrayParalleTask
Task.WaitAll(toGrayTasks)




