[<AutoOpen>]
module Piri.Actors

open Akka.Actor
open Akka.Routing
open Nessos.Vagabond
open Nessos.Vagabond.AssemblyProtocols
open Nessos.Vagabond.ExportableAssembly

type ActorContext = { Context: IActorContext }

/// Untyped version of Actor behavior used internally - since we are using 
/// DynamicActor no matter what message/state type its behavior is using
type Receive = ActorContext -> obj -> obj -> obj
/// Typed version of Actor behavior - for user convenience, exposed in the API
type Receive<'State, 'Msg> = ActorContext -> 'State -> 'Msg -> 'State

/// All informations necessary for Server (worker supervisor) to instantiate worker actors
type ActivationConfig =
    { // Name of the client
      Name: string
      // Current worker actors behavior
      Behavior: Receive
      // Initial state of each actor
      Zero: obj
      // Routing logic used to distribute messages send by client across the actors
      RoutingLogic: RoutingLogic
      // Number of actors per each cluster node
      ActorsPerNode: int }
    static member Create(name, behavior: Receive<'State, 'Msg>, zero, routingLogic, actorsPerNode) =
        // convert typed Receive defined by the user into untyped version
        let typeless: Receive = 
            fun ctx state msg -> 
                let typedState = downcast state
                let typedMsg = downcast msg
                upcast behavior ctx typedState typedMsg
        { Name = name
          Behavior = typeless
          Zero = zero
          RoutingLogic = defaultArg routingLogic (Akka.Routing.RoundRobinRoutingLogic() :> RoutingLogic)
          ActorsPerNode = defaultArg actorsPerNode 1 }

/// Messages send directly to DynamicActor
type ActorMessage =
    /// Reload actor's behavior with new Receive function
    | Load of Receive
    /// Stop an actor
    | Stop
    /// Get current actor's state (it will be send as response)
    | GetState

/// Messages send to a client
type ClientMessage =
    /// Upload dependencies using Nessos.Vagabond
    | UploadedDependencies of IRemoteAssemblyReceiver * DataDependencyInfo []
    /// Info from server to client about actors that have been created
    | ActorsLoaded of IActorRef []
    /// Request to reload behavior of all client's actors with new Receive
    | ReloadConfig of Receive

/// Messages send to a server
type ServerMessage =
    /// Request for list of assembly infos on current cluster node 
    /// (required by Vagabond to compute dependencies to send)
    | GetAssemblyInfo of AssemblyId []
    /// Request from client to load Vagabond exportable assemblies 
    /// as dynamic assemblies on current cluster node
    | LoadAssemblies of ExportableAssembly []
    /// Request from client to server to load actors using defined parameters
    | LoadActors of count:int * behavior:Receive * zero:obj * name:string
        
/// Actor with state and behavior (receive param) that can be switched in runtime
type DynamicActor(receive: Receive, zero: obj) as this =
    inherit UntypedActor()
    let untypedContext = UntypedActor.Context :> IActorContext
    let ctx = { Context = untypedContext}
    let mutable currentState: obj = zero
    let mutable currentReceive: Receive = receive
    override this.OnReceive(msg:obj) = 
        match msg with
        | :? ActorMessage as message ->
            match message with
            | GetState -> untypedContext.Sender.Tell currentState
            | Stop -> untypedContext.Stop(untypedContext.Self)
            | Load(newReceive) -> currentReceive <- newReceive
        | message -> currentState <- currentReceive ctx currentState message 

[<AutoOpen>]
module Actors =
    let (<!) (toRef: IActorRef) (msg: obj) = toRef.Tell(msg)
    let (<?) (toRef: IActorRef) (msg: obj) = toRef.Ask(msg) |> Async.AwaitTask