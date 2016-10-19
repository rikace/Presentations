module Piri.Server

open System
open Akka.FSharp
open Akka.Actor
open Nessos.Vagabond
open Nessos.Vagabond.AssemblyProtocols
open Nessos.Vagabond.ExportableAssembly

/// Server receive method (used to instantiate actor on the server node, 
/// which will supervise worker actors)
let receive (vm: VagabondManager) : (Actor<_> -> Cont<_, _>) =
    fun ctx -> 
        printfn "Created server actor: %s" (ctx.Self.ToString())
        // define actor loop with Map<string, IActorRef list> of actors
        // Map's key matches client's name, value is list of worker actors
        // for that particular client
        let rec loop actorsMap = actor {
            let! (msg) = ctx.Receive()
            match msg with
            | GetAssemblyInfo ids ->
                let infos = vm.GetAssemblyLoadInfo ids
                ctx.Sender() <! infos
                return! loop actorsMap
            | LoadAssemblies asm ->
                printfn "Loading assemblies"
                let vas = vm.CacheRawAssemblies asm
                let infos = vm.LoadVagabondAssemblies vas
                ctx.Sender() <! infos
                return! loop actorsMap
            | LoadActors (count, behavior, zero, name) ->
                // request from client to load actors
                let replyTo = ctx.Sender()
                let refs =
                    [| 1..count |]
                    |> Array.map(fun i ->
                        let childName = name + i.ToString()
                        let ref = 
                            // try to find worker ref, if it's not present, create one
                            match ctx.Context.Child(childName) with
                            | r when Object.Equals(r, ActorRefs.Nobody) ->
                                printfn "Creating agent: %s" childName
                                let props = Props.Create<DynamicActor>(behavior, zero)
                                ctx.ActorOf(props, childName)
                            | r -> r
                        // update worker's behavior
                        ref <! Load(behavior)
                        ref
                    )
                // send to client list of all worker refs (both existing and the new ones)                            
                replyTo <! ActorsLoaded refs
                return! loop (Map.add name refs actorsMap)
        }
        // start actor's receive loop
        loop Map.empty
