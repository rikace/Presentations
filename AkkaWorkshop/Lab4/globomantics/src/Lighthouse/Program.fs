// Learn more about F# at http://fsharp.org
// See the 'F# Tutorial' project for more help.

[<EntryPoint>]
let main argv = 
    use system = Akka.Actor.ActorSystem.Create("globomantics")
    system.WhenTerminated.Wait()
    0 // return an integer exit code
