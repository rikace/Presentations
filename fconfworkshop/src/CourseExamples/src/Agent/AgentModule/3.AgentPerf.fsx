#load "Utils.fs"
open System.IO
open System
open System.Threading
open System.Collections.Generic
open System.Net
open AgentModule

   
// ===========================================
// Agent perf message-sec
// ===========================================

let [<Literal>] count = 2000000

let agent() =
    Agent<int>.Start(fun inbox ->
            let sw = System.Diagnostics.Stopwatch()
            let rec loop() = async{
                let! msg = inbox.Receive()
                if msg = 0 then 
                    sw.Start()
                    return! loop()
                elif msg = count then
                    printfn "Last message arrived - %d ms - %d message per sec" sw.ElapsedMilliseconds (count/  sw.Elapsed.Seconds)
                else
                    return! loop() }
            loop())

let sw = System.Diagnostics.Stopwatch.StartNew()
let agents = Array.init count (fun _ -> agent())
printfn "Time to create %d Agents - %d ms" count sw.ElapsedMilliseconds
    
sw.Restart()    
agents |> Array.iteri(fun i a -> a.Post(i))

printfn "Last message sent - %d ms - %d message per sec" sw.ElapsedMilliseconds (count/  sw.Elapsed.Seconds)


