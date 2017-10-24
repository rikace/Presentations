module Consumer

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

module Actors =
    type PlateEvent = { totalBases:int; atBats:int; rbi:int; hits:int; walks:int; hitByPitch:int; sacrificeFlies:int }
  
    type ActorMetaData = { name:string; path:string}
        with static member Init (name:string) = { name=name; path=sprintf "/user/%s" name }

    let serializeEvent (o:obj) = 
       o |> (Newtonsoft.Json.JsonConvert.SerializeObject >> System.Text.Encoding.Default.GetBytes)
    
    let deserialize<'a> data =
        data |> (System.Text.Encoding.Default.GetString >> Newtonsoft.Json.JsonConvert.DeserializeObject<'a>)

    let average atBats hits = 
        if atBats = 0 then 0.
        else float hits / float atBats

    let slugging atBats totalBases = 
        if atBats = 0 then 0.
        else float totalBases / float atBats        

    let onBase hits walks hitByPitch atBats sacrificeFlies = 
        (float hits + float walks + float hitByPitch) / (float(atBats + walks + hitByPitch + sacrificeFlies))        
          
    module paths = 
        let gameEventCoordinator = ActorMetaData.Init "gameCoordinator"

    let batterActor (actorId:string) (mailbox : Actor<_>) =     
    
        let handleHitterAtPlateEvent (p:PlateEvent) (msg:HitterWasAtBat) =
            let totalBases = msg.hitValue + p.totalBases
            let atBats = if msg.isAtBat then 1 else 0
            let rbi = msg.rbiOnPlay + p.rbi
            let hits = if msg.hitValue > 0 then 1 else 0
            let walks =
                if enum msg.playType = PlayType.Walk then p.walks + 1 else p.walks
            let hitByPitch =
                if enum msg.playType = PlayType.HitByPitch then p.hitByPitch + 1 else p.hitByPitch
            let sacrificeFlies =
                if msg.isSacrificeFly then p.sacrificeFlies + 1 else p.sacrificeFlies
            { totalBases=totalBases; atBats=atBats; rbi=rbi; hits=hits; walks=walks; hitByPitch=hitByPitch; sacrificeFlies=sacrificeFlies }
            
        let rec loop state = actor { 
            let! (cmds:HitterWasAtBat list) = mailbox.Receive()
            printfn "Received %d HitterWasAtBat" (cmds |> Seq.length)
            let newState =
                cmds
                |> Seq.fold(fun acc msg -> 
                    let pevent = handleHitterAtPlateEvent acc msg
                    let batter = { Batter.atBats=pevent.atBats
                                   average=average pevent.atBats pevent.hits
                                   hits=pevent.hits
                                   batterId=actorId
                                   playerId=actorId
                                   name=""
                                   onBase=onBase pevent.hits pevent.walks pevent.hitByPitch pevent.atBats pevent.sacrificeFlies
                                   sacrificeFlies=pevent.sacrificeFlies
                                   slugging=slugging pevent.atBats pevent.totalBases
                                   walks=pevent.walks 
                                   hitByPitch=pevent.hitByPitch
                                   totalBases=pevent.totalBases 
                                }
                    printfn "%s has %f BA, %f OBP, and %f SLG" actorId batter.average batter.onBase batter.slugging

                    RethinkDB.R.Db("baseball").Table("batterStat").Insert(batter).RunNoReply(rethinkConnection.Value)
                    
                    //write to event store                    
                    let event = serializeEvent msg
                    let eventData = EventData(Guid.NewGuid(), "HitterPlateAppearance", true, event, Array.zeroCreate<byte> 1)

                    // (3)  Plug the EventStore here
                    //      Use the AppendToStreamAsync API to store the eventData
                    //      The computation is asynchronous (Task<_>), thus, use the "PipeTo" operator 
                    //      to send the output to the current mailbox recipient 

                    
                    pevent) state                    
            return! loop newState
        }
        loop { totalBases=0; atBats=0; rbi=0; hits=0; walks=0; hitByPitch=0; sacrificeFlies=0 }

    let pitcherActor (actorId:string) (mailbox : Actor<_>) =            
        let rec loop state = actor { 
            let! (msg:PitcherFacedBatter) = mailbox.Receive()
        
            let totalBases = state.totalBases + msg.hitValue
            let atBats = if msg.isAtBat then state.atBats + 1 else state.atBats
            let hits = if msg.hitValue > 0 then state.hits + 1 else state.hits
            let walks = if enum msg.playType = PlayType.Walk then state.walks + 1 else state.walks
            let hitByPitch = if enum msg.playType = PlayType.HitByPitch then state.hits + 1 else state.hits
            let sacrificeFlies = if msg.isSacrificeFly then state.sacrificeFlies + 1 else state.sacrificeFlies
            
            let pitcher = { Pitcher.atBats=atBats; oppAvg=average atBats hits; hits=hits; pictherId=actorId; name=""; oppObp=onBase hits walks hitByPitch atBats sacrificeFlies; sacrificeFlies=sacrificeFlies; oppSlugging=int(slugging atBats totalBases); walks=walks; totalBases=totalBases; hitByPitch=hitByPitch }

            let event = serializeEvent pitcher           
            let eventData = EventData(Guid.NewGuid(), "PitcherFacedBatter", true, event, Array.zeroCreate<byte> 1)
            
            eventConnection.Value.AppendToStreamAsync(msg.pictherId, ExpectedVersion.Any, eventData)
            |> Async.AwaitTask
            |!> mailbox.Self

            return! loop { state with totalBases=totalBases; atBats=atBats; walks=walks; hits=hits; sacrificeFlies=sacrificeFlies; hitByPitch=hitByPitch }
        }
        loop { PlateEvent.totalBases=0; atBats=0; rbi=0; hits=0; walks=0; hitByPitch=0; sacrificeFlies=0 }

    let pitcherSupervisor (mailbox : Actor<_>) = 
        let rec loop() = actor { 
                let! (msg:HandleNewGameEvent) = mailbox.Receive()
                let pitcherId = msg.pitcher
                let pitcherActorChild = 
                    let pitcherActorChild = mailbox.Context.Child(pitcherId)
                    if pitcherActorChild.IsNobody() then spawn mailbox pitcherId (pitcherActor pitcherId)
                    else pitcherActorChild
                let pictherFaceBatter = { pictherId=pitcherId; name=""; hitValue=msg.hitValue; isAtBat=msg.isAtBat; playType=msg.success; isSacrificeFly=msg.isSacFly }                
                pitcherActorChild <! pictherFaceBatter
                return! loop()
            }
        loop()
        
    let batterSupervisor (mailbox : Actor<_>) = 
        let rec loop() = actor { 
                let! (msg:HandleNewGameEvent) = mailbox.Receive()
                let batterId = msg.batter
                let batterActor =
                    let batterActorChild = mailbox.Context.Child(batterId)
                    if batterActorChild.IsNobody() then
                        let batterActor = spawn mailbox batterId (batterActor batterId)
                        
                        // get hitter events for this batter from event store
                        let rec computeEvents index = 
                            let eventSlice = eventConnection.Value.ReadStreamEventsForwardAsync(msg.batter, index, index + 100, false).Result
                            let events = eventSlice.Events |> Seq.map(fun e -> deserialize<HitterWasAtBat> e.Event.Data)
                            batterActor <! (events |> Seq.toList)
                            if eventSlice.IsEndOfStream then ()
                            else computeEvents (index + 101) 
                        computeEvents 0

                        batterActor
                    else batterActorChild
                let team = if msg.isHomeAtBat then msg.homeTeam else msg.visitingTeam
                let hitterWasAtBatMsg = { HitterWasAtBat.id=batterId; name=""; team=team; pitcherId=msg.pitcher; hitValue=msg.hitValue; rbiOnPlay=msg.rbiOnPlay; balls=msg.balls; strikes=msg.strikes; outs=msg.outs; inning=msg.inning; isAtBat=msg.isAtBat; isSacrificeFly=msg.isSacFly; playType=msg.success }
                batterActor <! hitterWasAtBatMsg
                return! loop()
            }
        loop()

    let gameActor (actorId:string) (gameDate:System.DateTime) (mailbox : Actor<_>) = 
        let rec loop() = actor { 
                let! (cmd:HandleNewGameEvent) = mailbox.Receive()
                let homeTeam = cmd.homeTeam
                let visitingTeam = cmd.visitingTeam
                let inning = cmd.inning
                let outs = cmd.outs
                let strikes = cmd.strikes
                let balls = cmd.balls
                let homeScore = cmd.homeScore
                let visitorScore = cmd.visitorScore
                let batterId = cmd.batter
                let pitcherId = cmd.pitcher
                printfn "%s :: %s vs %s :: %d - %d :: %d %d %d %d" (gameDate.ToShortDateString()) homeTeam visitingTeam homeScore visitorScore inning outs strikes balls
                
                if cmd.rbiOnPlay > 0 then
                    let teamId = if cmd.isHomeAtBat then homeTeam else visitingTeam
                    let runScoredMsg = { TeamRunScored.teamId = teamId; runs = cmd.rbiOnPlay }
                    mailbox.Context.ActorSelection(paths.gameEventCoordinator.path) <! runScoredMsg   
                return! loop()
            }
        loop()
    
    let gameSupervisor (mailbox : Actor<_>) = 
        let rec loop() = actor { 
            let! (cmd:HandleNewGameEvent) = mailbox.Receive()
            let gameId = cmd.gameId           
            let actorGameId = sprintf "game-%s" gameId
            let gameActor = 
                let gameActorChild = mailbox.Context.Child(actorGameId)
                if gameActorChild.IsNobody() then                    
                    let year = int(gameId.Substring(3, 4))
                    let month = int(gameId.Substring(7, 2))
                    let day = int(gameId.Substring(9, 2))
                    let gameDate = DateTime(year, month, day)                    
                    printfn "creating game actor : %s" actorGameId
                    spawn mailbox actorGameId (gameActor gameId gameDate)
                else gameActorChild
            
            let pitcherSuper =
                let pitcherSuper = mailbox.Context.Child("pitcherSuper")
                if pitcherSuper.IsNobody() then spawn mailbox "pitcherSuper" pitcherSupervisor                           
                else pitcherSuper

            // (2)  Implement a batterSuper Actor 
            //      if the current Context is not aware of the actor batterSupervisor, then spawn a new one
            //      otherwise return the batterSuper from the Context
            let batterSuper = 
                // remove the line below and add the Avtor implementtaion
                pitcherSuper


            gameActor <! cmd
            batterSuper <! cmd
            pitcherSuper <! cmd 
            return! loop() }
        loop()

    let gameCoordinator (mailbox : Actor<_>) = 
        let rec loop() = actor { 
            let! (cmd:HandleNewGameEvent) = mailbox.Receive()            
            let gameSuper = 
                let gameSuper = mailbox.Context.Child("gameSuper")
                if gameSuper.IsNobody() then spawn mailbox "gameSuper" gameSupervisor
                else gameSuper
            gameSuper <! cmd
            printfn "Batter %s against Pitcher %s with pitch sequence %s was a %d." cmd.batter cmd.pitcher cmd.sequence cmd.success
            return! loop() }
        loop()

[<EntryPoint>]
let main argv =  
    let config = """
        akka {  
            log-config-on-start = on
            stdout-loglevel = DEBUG
            loglevel = DEBUG
          
            actor {
                provider = "Akka.Remote.RemoteActorRefProvider, Akka.Remote"
            }    
            remote.helios.tcp {
                transport-protocol = tcp
                port = 9234                 
                hostname = localhost  
                }
        }
        """

    System.Console.Title <- "Consumer : " + System.Diagnostics.Process.GetCurrentProcess().Id.ToString()

    let system = System.create "consumer-baseballStats" (Configuration.parse config)
    
    Console.ForegroundColor <- ConsoleColor.Green
    printfn "Actor-System %s listening..." system.Name

    let supervision = 
        Strategy.OneForOne(fun e ->
            match e with
            | _ -> Directive.Restart )

    // (1)  implement consumer Actor that receives messages from producer.
    //      apply the akka-router option to spawn multiple routees 
    //          using Routing (Round-Robin) spawn-option  (check function spawnOpt)
    //      Option : add Supervisor-Strategy
    //
    //      Note : the name of the actor will be used by the producer to push messages
    let router =  " < CODE HERE > " // check Akka.Routing  
    let consumerActor =  () // " < CODE HERE >  

    Console.ReadLine() |> ignore
    system.Terminate().Wait()
    
    0
