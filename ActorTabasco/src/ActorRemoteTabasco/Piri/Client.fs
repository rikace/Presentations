module Piri.Client

open System
open Akka.FSharp
open Akka.Actor
open Akka.Routing
open Akka.Cluster
open Nessos.Vagabond
open Nessos.Vagabond.ExportableAssembly
open Nessos.Vagabond.AssemblyProtocols

/// State of the client's actor
type internal ClientState = 
    { /// List of all connected Vagabond assembly receiver (one per each server node)
      Receivers: Set<AssemblyReceiver>
      /// List of all workers actors on all nodes
      Agents: Set<IActorRef>
      /// Configuration info necessary to create a worker actor
      Config: ActivationConfig
      // Buffer for the messages in case, when there are no worker actors yet
      MessageBuffer: (obj * IActorRef) list }
    static member Create(conf: ActivationConfig) =
        { Receivers = Set.empty
          Agents = Set.empty
          Config = conf
          MessageBuffer = [] }
    member x.AddAgents (refs) = { x with Agents = x.Agents + refs }
    member x.RemoveAgents (refs) = { x with Agents = x.Agents - refs }
    member x.AddReceiver (receiver) = { x with Receivers = Set.add receiver x.Receivers }
    member x.RemoveReceivers (receivers) = { x with Receivers = x.Receivers - receivers }
    member x.BufferMessage msg = { x with MessageBuffer = msg::x.MessageBuffer }
    member x.ClearBuffer() = { x with MessageBuffer = [] }

/// Trigger Vagabond protocol for hot loading code on the server nodes
let private loadDependencies (vmmanager: VagabondManager) conf replyTo (receiver: AssemblyReceiver) =
    async {
        let! deps = vmmanager.SubmitObjectDependencies(receiver, conf, permitCompilation = true)
        printfn "Submitted %d dependencies to %A" deps.Length receiver.ServerRef
        // inform current client once everything is set
        replyTo <! UploadedDependencies(receiver, deps)
    } |> Async.Start

/// Flush buffer back to the current client
let private flushBuffer state (recipent: IActorRef) =
    // since buffered messages are prepended, we need to flush them in reverse
    state.MessageBuffer
    |> List.rev
    |> List.iter (fun (msg, sender) -> recipent.Tell(msg, sender))
    state.ClearBuffer()

/// Client's main receive behavior
let internal receive (vmmanager: VagabondManager) (initState:ClientState) (ctx: Actor<obj>) =
    // invoke Akka.Cluster plugin and subscribe to it in order
    // to receive updates about the cluster state
    let cluster = Cluster.Get(ctx.Context.System)
    cluster.Subscribe(ctx.Self, [| typeof<ClusterEvent.IMemberEvent> |])
    // unsubscribe from cluster updates once current client dies
    ctx.Defer (fun () -> cluster.Unsubscribe ctx.Self)

    let rec loop (state: ClientState) = actor {
        let! msg = ctx.Receive()
        match msg with
        | :? ClusterEvent.MemberUp as up ->
            // when new node joins the cluster, every node will receive MemberUp event
            let addr = up.Member.UniqueAddress.Address.ToString()
            // try to find Piri server actor on the node that has joined,
            // ask it to identify itself
            let selection = ctx.ActorSelection (addr + "/user/piri-server")
            printfn "Sending identify request to %s" addr
            selection <! Identify(addr)
            return! loop state
        | :? ClusterEvent.MemberRemoved as removed ->
            // when existing node gets removed from the cluster, every node will receive MemberRemoved event
            let addr = removed.Member.Address
            // remove Vagabond assembly receiver used to hot load code on the removed server's node
            let removedReceiver = 
                state.Receivers
                |> Set.filter (fun r -> r.ServerRef.Path.Address = addr)
            return! loop (state.RemoveReceivers removedReceiver)
        | :? ClusterEvent.IMemberEvent -> return! loop state            // ignore other cluster events
        | :? ClusterEvent.CurrentClusterState -> return! loop state     // ignore cluster state updates
        | :? ActorIdentity as identity when identity.Subject <> null ->
            // when server ref has successfully identified itself, it sends back ActorIdentity

            // watch over server's ref to get acknowledged in case if it dies
            let ref = ctx.Watch identity.Subject
            printfn "Discovered Piri server at %s" (ref.Path.Address.ToString())
            // create Vagabond assembly receiver for the new node
            let receiver = AssemblyReceiver(vmmanager, ref)
            // order to load dependencies for behavior necessary by current client's workers
            loadDependencies vmmanager (state.Config.Behavior, state.Config.Zero) ctx.Self receiver
            return! loop (state.AddReceiver receiver)
        | :? ActorIdentity -> return! loop state        // ignore nodes that don't have Piri server plugin set up
        | :? Terminated as terminated ->
            // when watched actor dies, it sends back Terminated message
            
            // check if dead actor is one of the worker supervisors
            // if no, it was worker -> remove it from workers set
            // if yes, it was Vagabond assembly receiver -> remove it from receivers list
            let deadReceivers = 
                state.Receivers 
                |> Set.filter (fun r -> r.ServerRef = terminated.ActorRef)
            let updatedState =
                if deadReceivers.IsEmpty
                then state.RemoveAgents (Set.singleton terminated.ActorRef)
                else state.RemoveReceivers deadReceivers
            return! loop updatedState
        | :? ClientMessage as clientMsg ->
            match clientMsg with
            | UploadedDependencies(receiver, deps) ->
                // sent by the server actor when Vagabond dependencies has been uploaded
                let serverRef = (receiver :?> AssemblyReceiver).ServerRef
                printfn "Server at %s uploaded dependencies" (serverRef.Path.Address.ToString())
                let { Behavior = behavior; ActorsPerNode = count; Zero = zero; Name = name} = state.Config
                // request server to create worker actors
                serverRef <! LoadActors(count, behavior, zero, name)
                return! loop state
            | ActorsLoaded refs ->
                // once server created worker actors it replies with ActorsLoaded message
                printfn "Server %s has loaded %d actors" (ctx.Sender().Path.Address.ToString()) refs.Length
                // order client to watch over workers and update it's state
                let updatedState = 
                    refs
                    |> Array.map ctx.Watch
                    |> Set.ofArray
                    |> state.AddAgents
                // flush message buffer if there are workers able to process messages
                if not updatedState.Agents.IsEmpty
                then return! loop (flushBuffer updatedState ctx.Self)
                else return! loop updatedState
            | ReloadConfig behavior ->
                // reload workers behavior
                logInfo ctx "Config reload triggered"
                let conf = { state.Config with Behavior = behavior }
                // load all new code dependencies introduced by new behavior
                // using Vagabond assembly receivers to all known nodes
                state.Receivers
                |> Set.iter (loadDependencies vmmanager (behavior, conf.Zero) ctx.Self)
                // clean the agents - 
                // they will be reloaded as part of UploadedDependencies handler
                return! loop { state with Config = conf; Agents = Set.empty }
        | agentMsg ->            
            // if there are no agents listening, buffer message
            if state.Agents.IsEmpty
            then 
                printfn "Buffering message %A" agentMsg
                return! loop (state.BufferMessage (agentMsg, ctx.Sender()))
            else 
                // otherwise use routing logic to determine which of the workers should
                // get the message
                let routees = 
                    state.Agents 
                    |> Set.toArray
                    |> Array.map (fun a -> ActorRefRoutee(a) :> Routee)
                let agent = state.Config.RoutingLogic.Select(agentMsg, routees)
                printfn "Forwarding message %A to %A" agentMsg agent
                agent.Send(agentMsg, ctx.Sender())
                return! loop state
    }

    loop initState

/// Change typed behavior into ReloadConfig message with untyped behavior
let reload behavior = 
    let typeless: Piri.Actors.Receive = fun ctx state msg -> upcast behavior ctx (downcast state) (downcast msg) 
    ReloadConfig typeless