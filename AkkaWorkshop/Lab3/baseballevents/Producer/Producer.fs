module Producer

open Akka.FSharp
open Akka.Actor
open Akka.Remote
open Akka.Configuration
open RethinkDb.Driver
open RethinkDb.Driver
open RethinkDb.Driver.Ast
open RethinkDb.Driver.Model
open EventStore.ClientAPI
open System.Runtime.Serialization
open System.Reactive.Concurrency
open System.Reactive.Linq
open System.IO
open System.Threading.Tasks
open System.Threading
open System
open System.Net
open CommonTypes

[<Literal>]
let IP = "192.168.99.100"
let rethinkConnection = 
    lazy (
            RethinkDB.R.Connection().Hostname(IP).Port(RethinkDBConstants.DefaultPort).Timeout(30).Connect()
    )  

let eventConnection =
    lazy (  
            let settings = ConnectionSettings.Create().UseConsoleLogger().Build()
            let connection = EventStoreConnection.Create(settings, Uri(sprintf "tcp://admin:changeit@%s:1113" IP))        
            let task = connection.ConnectAsync() 
            task.Wait()
            connection
    )

module Observable =          
    let subscribeOn (scheduler:System.Reactive.Concurrency.IScheduler) (observable : IObservable<'T>) =
        observable.SubscribeOn(scheduler)


[<EntryPoint>]
let main argv = 
    let config =
        Configuration.parse
            @"akka {
                actor.provider = ""Akka.Remote.RemoteActorRefProvider, Akka.Remote""
                remote.helios.tcp {
                    hostname = localhost
                    port = 9238
                }
            }"

    let system = System.create "baseballStats" config

    System.Console.Title <- "Producer : " + System.Diagnostics.Process.GetCurrentProcess().Id.ToString()
    Console.ForegroundColor <- ConsoleColor.Green
    printfn "Actor-System %s listening..." system.Name
   
    // (1)  Refactor the Actor to forward the incoming message
    //      to the remote "consumerActor" Actor running in the Consumer project.
    //      Note : detect the remote address using the F# API select function
    let producerActor (mailbox : Actor<_>) = 
        let rec loop() = actor { 
            let! (cmd:HandleNewGameEvent) = mailbox.Receive()                        
            
            printfn "Sending new event to consumer : Batter %s against Pitcher %s with pitch sequence %s was a %d." cmd.batter cmd.pitcher cmd.sequence cmd.success
            // (-) Option: plug EventStore here
            return! loop() }
        loop()

    let consumerRemoteActor = select "akka.tcp://consumer-baseballStats@localhost:9234/user/gameCoordinator" system
    
    // (2)  update the actor accordingly with (1)
    let producer = spawn system "producer" producerActor
    
    let dbTable = RethinkDB.R.Db("baseball").Table("plays")   
    let feeds = dbTable.Changes().RunChanges<Play>(rethinkConnection.Value)
   
    use feedObs = 
        feeds.ToObservable() 
        // Observable subscribe on Task scheduler to run in parallel
        |> Observable.subscribeOn(CurrentThreadScheduler.Instance)
        |> Observable.subscribe(fun (p:Change<Play>) -> 
                                    let nv = p.NewValue
                                    let cmd = { HandleNewGameEvent.id = nv.id
                                                HandleNewGameEvent.gameId = nv.gameId
                                                homeTeam = nv.homeTeam
                                                success = nv.success
                                                visitingTeam = nv.visitingTeam
                                                sequence = nv.sequence
                                                inning = nv.inning
                                                balls = nv.balls
                                                strikes = nv.strikes
                                                outs = nv.outs
                                                homeScore = nv.homeScore
                                                visitorScore = nv.visitorScore
                                                rbiOnPlay = nv.rbiOnPlay
                                                hitValue = nv.hitValue
                                                batter = nv.batter
                                                pitcher = nv.pitcher
                                                isBatterEvent = nv.isBatterEvent
                                                isAtBat = nv.isAtBat
                                                isHomeAtBat = nv.isHomeAtBat
                                                isEndGame = nv.isEndGame
                                                isSacFly = nv.isSacFly
                                             }
                                    printfn "id change : %s" nv.gameId
                                  
                                    // (3)  send the "cmd"" message to the producer Actor (2)
                                    
                                    )
                                       
    Console.ReadLine() |> ignore
    system.Terminate().Wait()
    
    0