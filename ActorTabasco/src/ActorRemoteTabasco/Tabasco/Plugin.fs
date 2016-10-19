namespace Piri

open System
open System.IO
open System.Reflection
open Akka.Actor
open Akka.FSharp
open Akka.Routing
open Nessos.Vagabond

/// Akka's extension for current feature
type PiriPlugin (system: ExtendedActorSystem) =

    let config = ServerConfig.Create system
    // init Vagabond manager
    let vmmanager =
        if (Directory.Exists config.AssemblyCachePath) then
            Directory.Delete (config.AssemblyCachePath, true)
        Directory.CreateDirectory (config.AssemblyCachePath) |> ignore
        Vagabond.Initialize(
            cacheDirectory = config.AssemblyCachePath, 
            ignoredAssemblies = (config.KnownAssemblies |> List.map Assembly.Load),
            forceLocalFSharpCore = true)

    do
        // init all serializers required by Vagabond
        Nessos.FsPickler.FsPickler.RegisterPickler(actorRefPickler system)
        let actorSerializer = VagabondSerializer<ActorMessage>(system, vmmanager, 1337)
        let clientSerializer = VagabondSerializer<ClientMessage>(system, vmmanager, 1338)
        let serverSerializer = VagabondSerializer<ServerMessage>(system, vmmanager, 1339)
        system.Serialization.AddSerializer(actorSerializer)
        system.Serialization.AddSerializer(clientSerializer)
        system.Serialization.AddSerializer(serverSerializer)
        system.Serialization.AddSerializationMap(typeof<ActorMessage>, actorSerializer)
        system.Serialization.AddSerializationMap(typeof<ClientMessage>, clientSerializer)
        system.Serialization.AddSerializationMap(typeof<ServerMessage>, serverSerializer)

    interface IExtension

    /// Plugin configuration
    member x.Config = config
    /// Vagabond manager used for uploading assembly dependencies
    member x.VagabondManager = vmmanager
    /// Spawns an actor with behavior of tabasco host under curent /user root guardian
    member x.Server = spawn system "piri-server" (Server.receive vmmanager)
    /// Spawns an actor with behavior of tabasco client under curent /user root guardian
    member x.Client(name, behavior, zero, ?routingLogic, ?actorsPerNode) = 
        let activationConfig = ActivationConfig.Create(name, behavior, zero, routingLogic, actorsPerNode)
        let clientConfig = Client.ClientState.Create(activationConfig)
        spawn system "piri-client" (Client.receive vmmanager clientConfig)

    /// Returns a tabasco plugin.
    static member Get(s: ActorSystem) = 
        s.WithExtension<PiriPlugin, ExtensionProvider>()

and internal ExtensionProvider () =
    inherit ExtensionIdProvider<PiriPlugin>()
    override x.CreateExtension (system) = PiriPlugin(system)